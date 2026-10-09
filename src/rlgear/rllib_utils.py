import csv
import logging
import numbers
import random
import re
import shutil
import string
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import ray
import ray.rllib.algorithms.callbacks
import ray.tune
import ray.tune.registry
import ray.tune.trainable.trainable
import ray.tune.utils
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.experiment.trial import Trial

logger = logging.getLogger(__name__)

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = object


def make_callbacks(params: dict[str, Any]) -> list[ray.tune.Callback]:

    callbacks: list[ray.tune.Callback] = []
    filt = Filter(params.get("excludes", []))
    if "csv" in params:
        callbacks.append(
            CSVFilteredLoggerCallback(filt, params["csv"]["wait_iterations"])
        )

    if "tensorboard" in params:
        callbacks.append(
            TBXFilteredLoggerCallback(filt, params["tensorboard"].get("prefixes", {}))
        )

    if "json" in params:
        callbacks.append(JsonFiltredLoggerCallback(filt))

    return callbacks


class Filter:
    """Filter extra :class:`ray.tune.logger.LoggerCallback` outputs.

    Parameters
    ----------
    excludes: list[str]
        list of regexes to be compiled via :func:`re.compile`

    """

    def __init__(self, excludes: list[str]):
        self.regexes = [re.compile(e) for e in excludes]

    def __call__(self, d: dict[str, Any]) -> dict[str, Any]:
        flat_result = ray.tune.utils.flatten_dict(d, delimiter="/")

        out = {}

        for key, val in flat_result.items():
            if not any(regex.match(key) for regex in self.regexes):
                out[key] = val

        return out


class AdjPrefix:
    def __init__(self, prefixes: dict[str, str]):
        self.prefixes = prefixes

    def adj(self, val: str) -> str:
        for old_prefix, new_prefix in self.prefixes.items():
            if val.startswith(old_prefix):
                return val.replace(old_prefix, new_prefix, 1)

        return val


class SummaryWriterAdjPrefix(SummaryWriter):
    def __init__(self, prefixes: dict[str, str], *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.adj_prefix = AdjPrefix(prefixes)

    def add_scalar(self, tag: str, *args: Any, **kwargs: Any) -> None:
        new_tag = self.adj_prefix.adj(tag)
        return super().add_scalar(new_tag, *args, **kwargs)


class TBXFilteredLoggerCallback(ray.tune.logger.tensorboardx.TBXLoggerCallback):
    """Wrap :class:`ray.tune.logger.tensorboardx.TBXLoggerCallback`.

    Reduces the output based on the provided :func:`Filter`.
    """

    def __init__(self, filt: Filter, prefixes: dict[str, str]):
        super().__init__()
        self.filt = filt
        self.prefixes = prefixes
        self._summary_writer_cls = self._summary_writer_cls_rm_prefix

    def log_trial_result(
        self, iteration: int, trial: Trial, result: dict[str, Any]
    ) -> None:
        super().log_trial_result(iteration, trial, self.filt(result))

    def _summary_writer_cls_rm_prefix(
        self, *args: Any, **kwargs: Any
    ) -> SummaryWriterAdjPrefix:
        return SummaryWriterAdjPrefix(self.prefixes, *args, **kwargs)


class JsonFiltredLoggerCallback(ray.tune.logger.json.JsonLoggerCallback):
    """Wrap :class:`ray.tune.logger.json.JsonLoggerCallback`.

    Reduces the output based on the provided :func:`Filter`.
    """

    def __init__(self, filt: Filter):
        super().__init__()
        self.filt = filt

    def log_trial_result(
        self, iteration: int, trial: Trial, result: dict[str, Any]
    ) -> None:
        super().log_trial_result(iteration, trial, self.filt(result))


class CSVFilteredLoggerCallback(ray.tune.logger.csv.CSVLoggerCallback):
    """Wrapper around :class:`ray.tune.logger.csv.CSVLoggerCallback` \
        that reduces the output based on the provided excludes regexes.

    This callback also waits a set number of training iterations before
    freezing the keys as sometimes not all logging items are available
    on the first iteration.
    """

    def __init__(self, filt: Filter, wait_iterations: int):
        super().__init__()
        self.wait_iterations = wait_iterations
        self.prior_results: dict[Trial, list[dict[str, Any]]] = defaultdict(list)
        self.keys: dict[Trial, set[str]] = defaultdict(set)
        self.filt = filt
        self.excluded_keys: set[str] = set()

    def log_trial_result(
        self, iteration: int, trial: Trial, result: dict[str, Any]
    ) -> None:

        # see piece of ray.tune.logger.csv.CSVLoggerCallback:log_trial_result
        if trial not in self._trial_files:
            self._setup_trial(trial)

        training_iteration = result["training_iteration"]
        result = self.filt(result)
        self.excluded_keys.update(
            {k for k, r in result.items() if not isinstance(r, numbers.Real)}
        )
        self.keys[trial] |= set(result)

        if not self._trial_csv[trial] and training_iteration >= self.wait_iterations:

            keys = self.keys[trial] - self.excluded_keys

            # see piece of ray.tune.logger.csv.CSVLoggerCallback:log_trial_result
            self._trial_csv[trial] = csv.DictWriter(self._trial_files[trial], keys)
            if not self._trial_continue[trial]:
                self._trial_csv[trial].writeheader()

            # now that we have a csv, write the cached results
            for r in self.prior_results[trial]:
                self.writerow(self._trial_csv[trial], r)

        if self._trial_csv[trial]:
            self.writerow(self._trial_csv[trial], result)
            self._trial_files[trial].flush()
        else:
            # We don't have all the keys yet so don't want to write a header.
            # For now cache the results.
            self.prior_results[trial].append(result)

    @staticmethod
    def writerow(csv_writer: csv.DictWriter, result: dict[Any, Any]) -> None:
        # copied from piece of ray.tune.logger.csv.CSVLoggerCallback:log_trial_result
        csv_writer.writerow({k: result.get(k, np.nan) for k in csv_writer.fieldnames})


def gen_passwd(size: int) -> str:
    """Generate password for ray.init call.

    This function was adapted from https://stackoverflow.com/a/2257449

    Example
    -------

    .. code-block:: python

      ray.init(redis_password=gen_passwd(512))

    Parameters
    ----------
    size : int
        how long the password should be

    """
    # https://stackoverflow.com/a/2257449
    chars = string.ascii_letters + string.digits
    return "".join(random.SystemRandom().choice(chars) for _ in range(size))


class RllibSaver:
    def __init__(self, interval: int, log_dir: Path, max_num: int):
        self.interval = interval
        self.log_dir = log_dir
        self.max_num = max_num

        self.last_elapsed = 0
        self.save_paths: list[Path] = []

    def save(self, elapsed: int, alg: Algorithm, force: bool) -> Optional[Path]:
        if not force and elapsed - self.last_elapsed < self.interval:
            return None

        self.last_elapsed = elapsed

        while len(self.save_paths) > self.max_num - 1:
            path = self.save_paths.pop(0)
            logger.debug(f"removing {path}")
            shutil.rmtree(path)

        save_path_inp = self.log_dir / f"ckpts/{elapsed:06d}"
        save_path = alg.save_to_path(save_path_inp)
        self.save_paths.append(save_path)
        logger.info(f"saved checkpoint to {save_path}")
        return save_path
