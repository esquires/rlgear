import copy
import pprint
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


class Curriculum:
    def __init__(
        self,
        prior_episodes: int,
        idx: int,
        init_setting: dict[str, Any],
        spec: list[dict[str, Any]],
    ):
        """

        prior_episodes : int
            the number of episodes to use for evaluating criteria for
            increasing the curriculum index
        idx : int
            the current curriculum index
        init_setting: dict[str, Any]
            how the curriculum should begin. For example, in an environment
            where the agent starts at "init_dist" from a goal, you can run::

            init_setting = {"init_dist": 1}

        spec: list[dict[str, Any]]
            each entry in the spec is a phase of the curriculum. You can specify
            changes within this phase. For instance, if you wanted "init_dist" to increase
            by 0.1 every step of the curriculum up to 10 and wanted
            to increment the curriclum index when the "final_dist" is less than 0.5,
            you can run::

            spec = [
                {
                    "name": "dist",
                    "thresh": {"final_dist": ["<", 0.5]},
                    "setting": {"init_dist": 1.1},
                    "delta": {"init_dist": 0.1},
                    "delta_count": 89,
                    "lim": {"init_dist": 10}
                }
            ]
        """
        self.prior_episodes = prior_episodes
        self.idx = idx
        self.info_hx: dict[str, list[float]] = {}

        self.thresholds: list[dict[str, tuple[str, float]]] = []
        self.settings: list[dict[str, float]] = [copy.deepcopy(init_setting)]
        self.names = ["init"]

        for spec_dict in spec:

            if not spec_dict:
                break

            base_thresh = spec_dict["thresh"]

            base_setting = spec_dict["setting"]

            name = spec_dict.get("name", "n/a")
            self.names.append(name)
            self.thresholds.append(base_thresh)
            self.settings.append(base_setting)

            for j in range(spec_dict.get("delta_count", 0)):
                base_thresh = copy.deepcopy(base_thresh)
                base_setting = copy.deepcopy(base_setting)

                for key, val in spec_dict.get("delta", {}).items():
                    if key in base_thresh:
                        base_thresh[key][1] += val
                        self._check_lim(spec_dict, key, base_thresh[key][1], val > 0, j)

                    if key in base_setting:
                        base_setting[key] += val
                        self._check_lim(spec_dict, key, base_setting[key], val > 0, j)

                self.names.append(name)
                self.thresholds.append(base_thresh)
                self.settings.append(base_setting)

    def update(self, info: dict[str, float]) -> bool:
        """check the current curriculum has been satisfied and increase the index.

        info : dict[str, float]
            a dictionary with keys matching Curriculum.thresholds[idx].
            Values will be copied into a list and when
            all lists meet the threshold requirements, the index will be incremented.
            As an example, with::

                self.thresholds[self.idx] = {"final_dist": ["<", 0.5]},

            you could set info as follows::

                info = {"final_dist": 0.4}

            After calling this function self.prior_episodes times, the index
            will increment.
        """
        for key, value in info.items():
            if key not in self.info_hx:
                self.info_hx[key] = []

            try:
                if not np.isnan(value):
                    self.info_hx[key].append(value)
            except:
                import lvdb; lvdb.set_trace()  # fmt: skip

        if self.idx >= len(self.settings) - 1:
            return False

        satisfies_thresholds = True

        for key, (comparator, thresh) in self.thresholds[self.idx].items():
            if key not in self.info_hx:
                satisfies_thresholds = False
                break

            hx = self.info_hx[key][-self.prior_episodes :]
            if len(hx) < self.prior_episodes:
                satisfies_thresholds = False
                break

            mean_hx = np.mean(hx).item()

            if (comparator == "<" and mean_hx >= thresh) or (
                comparator == ">" and mean_hx <= thresh
            ):
                satisfies_thresholds = False
                break

        if satisfies_thresholds:
            self.idx += 1
            self.info_hx = {}
            return True
        else:
            return False

    def to_dataframe(self) -> pd.DataFrame:

        def _get_dict_keys(_list: list[dict[str, Any]]) -> dict[str, list[Any]]:
            # note: using dict to preserve insertion order
            _names: dict[str, list[Any]] = {}
            for _l in _list:
                for _key in _l:
                    if _key not in _names:
                        _names[_key] = []
            return _names

        threshold_names = _get_dict_keys(self.thresholds)
        settings_names = _get_dict_keys(self.settings)

        no_val = "-"

        data: dict[str, list[Any]] = {"idx": [], "name": [], "subidx": []}
        data = {**data, **settings_names, **threshold_names}

        # pylint: disable=consider-using-enumerate
        for i in range(len(self.settings)):
            data["name"].append(self.names[i])
            data["idx"].append(i)

            if i == 0 or self.names[i] != self.names[i - 1]:
                subidx = 0
            else:
                subidx = data["subidx"][-1] + 1

            data["subidx"].append(subidx)

            for name in settings_names:
                val = self.settings[i][name] if name in self.settings[i] else no_val
                data[name].append(val)

            for name in threshold_names:
                if i < len(self.thresholds) and name in self.thresholds[i]:
                    thresh = self.thresholds[i][name]
                    val = f"{thresh[0]} {thresh[1]}"
                else:
                    val = no_val
                data[name].append(val)

        df = pd.DataFrame.from_dict(data)
        df.reset_index(drop=True)
        return df

    @staticmethod
    def _check_lim(
        spec_dict: dict[str, Any], key: str, val: float, is_pos_delta: bool, j: int
    ) -> None:

        if "lim" not in spec_dict or key not in spec_dict["lim"]:
            return

        lim = spec_dict["lim"][key]
        if (val < lim and not is_pos_delta) or (val > lim and is_pos_delta):
            msg = f'"{key}" has a limit of {lim} but reaches a value of {val}. '
            if j > 0:
                msg += f"Perhaps set delta_count = {j} in the below spec\n"
            else:
                msg += "The particular spec is\n"
            msg += pprint.pformat(spec_dict)
            raise RuntimeError(msg)
