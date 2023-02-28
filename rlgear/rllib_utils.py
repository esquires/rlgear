import argparse
import collections
import re
import csv
import os
import string
import random
from typing import Any, Union, Dict, Set, List

import numpy as np
import torch

import ray
import ray.tune.utils
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2

from .utils import MetaWriter, StrOrPath, import_class


class MetaLoggerCallback(ray.tune.logger.LoggerCallback):
    def __init__(self, meta_writer: MetaWriter):
        self.meta_writer = meta_writer

    def log_trial_start(self, trial: ray.tune.experiment.trial.Trial) -> None:
        self.meta_writer.write(trial.logdir)


class Filter:
    def __init__(self, excludes: List[str]):
        self.regexes = [re.compile(e) for e in excludes]

    def __call__(self, d: Dict[str, Any]) -> Dict[str, Any]:
        flat_result = ray.tune.utils.flatten_dict(d, delimiter="/")

        out = {}

        for key, val in flat_result.items():
            if not any(regex.match(key) for regex in self.regexes):
                out[key] = val

        return out


class TBXFilteredLoggerCallback(
            ray.tune.logger.tensorboardx.TBXLoggerCallback):
    def __init__(self, filt: Filter):
        super().__init__()
        self.filt = filt

    def log_trial_result(
        self,
        iteration: int,
        trial: ray.tune.experiment.trial.Trial,
        result: Dict[str, Any]
    ) -> None:
        super().log_trial_result(iteration, trial, self.filt(result))


class JsonFiltredLoggerCallback(ray.tune.logger.json.JsonLoggerCallback):
    def __init__(self, filt: Filter):
        super().__init__()
        self.filt = filt

    def log_trial_result(
        self,
        iteration: int,
        trial: ray.tune.experiment.trial.Trial,
        result: Dict[str, Any]
    ) -> None:
        super().log_trial_result(iteration, trial, self.filt(result))


class CSVFilteredLoggerCallback(ray.tune.logger.csv.CSVLoggerCallback):
    """Filter results as well as wait to get all possible logging items."""

    def __init__(self, wait_iterations: int, excludes: List[str]):
        super().__init__()
        self.wait_iterations = wait_iterations
        self.prior_results: Dict[Any, Any] = collections.defaultdict(list)
        self.keys: Set[str] = set()
        self.excludes = excludes

    def log_trial_result(
        self,
        iteration: int,
        trial: ray.tune.experiment.trial.Trial,
        result: Dict[str, Any]
    ) -> None:
        if trial not in self._trial_files:
            self._setup_trial(trial)

        tmp = result.copy()
        tmp.pop("config", None)
        result = ray.tune.utils.flatten_dict(tmp, delimiter="/")

        self.keys |= set(result)

        if not self._trial_csv[trial] and \
                result['training_iteration'] >= self.wait_iterations:
            # filter the results: explict excludes as well as those that are
            # lists
            keys = set()
            regexes = [re.compile(e) for e in self.excludes]
            for key in self.keys:
                if not any(regex.match(key) for regex in regexes) and \
                        not any(isinstance(res.get(key, np.nan), (str, list))
                                for res in self.prior_results[trial]):
                    keys.add(key)

            self._trial_csv[trial] = csv.DictWriter(
                self._trial_files[trial], keys
            )
            if not self._trial_continue[trial]:
                self._trial_csv[trial].writeheader()

            for r in self.prior_results[trial]:
                self._trial_csv[trial].writerow(
                    {k: r.get(k, np.nan)
                     for k in self._trial_csv[trial].fieldnames})

        if self._trial_csv[trial]:
            self._trial_csv[trial].writerow(
                {k: result.get(k, np.nan)
                 for k in self._trial_csv[trial].fieldnames})
            self._trial_files[trial].flush()
        else:
            self.prior_results[trial].append(result)


def add_rlgear_args(parser: argparse.ArgumentParser) \
        -> argparse.ArgumentParser:
    parser.add_argument('yaml_file')
    parser.add_argument('exp_name')
    parser.add_argument('--overrides')
    parser.add_argument('--debug', action='store_true')
    return parser


# pylint: disable=too-many-branches
def make_tune_kwargs(
    params: dict[Any, Any],
    meta_writer: MetaWriter,
    log_dir: StrOrPath,
    debug: bool,
) -> dict[str, Any]:
    # for inputs, see rlgear.utils.from_yaml

    kwargs: dict[str, Any] = {
        'config': {"log_level": "INFO"},
        'local_dir': str(log_dir),
    }

    for blk in params['rllib']['tune_kwargs_blocks'].split(','):
        kwargs = ray.tune.utils.merge_dicts(kwargs, params['rllib'][blk])

    cfg = kwargs['config']

    if debug:
        kwargs['local_dir'] = os.path.join(kwargs['local_dir'], 'debug')
        cfg['num_workers'] = 0
        cfg['num_gpus'] = 0
        kwargs['max_failures'] = 0
        kwargs['num_samples'] = 1
        if kwargs['verbose'] == 0:
            kwargs['verbose'] = 1
        if cfg['log_level'] in ['ERROR', 'WARN']:
            cfg['log_level'] = 'INFO'

    # handle the rllib logger callbacks. These are more complicated because
    # they need to be classes, not the objects. For now just handling
    # the case of a single callback but if needed in the future add support
    # for callbacks.MultiCallbacks
    if 'callbacks' in cfg:
        if isinstance(cfg['callbacks'], str):
            cfg['callbacks'] = import_class(cfg['callbacks'])
        else:
            cfg['callbacks'] = ray.rllib.algorithms.callbacks.MultiCallbacks(
                [import_class(cb) for cb in cfg['callbacks']])

    # handle the tune logger callbacks
    if 'callbacks' not in kwargs:
        kwargs['callbacks'] = []
    else:
        for i, cb in enumerate(kwargs['callbacks']):
            if isinstance(cb, dict):
                kwargs['callbacks'][i] = import_class(kwargs['callbacks'][i])

    if 'csv' in params:
        kwargs['callbacks'].append(CSVFilteredLoggerCallback(
            params['csv']['wait_iterations'], params['csv']['excludes']))

    if 'tensorboard' in params:
        kwargs['callbacks'].append(TBXFilteredLoggerCallback(
            Filter(params['tensorboard']['excludes'])))

    if 'json' in params:
        kwargs['callbacks'].append(JsonFiltredLoggerCallback(
            Filter(params['json']['excludes'])))

    kwargs['callbacks'].append(MetaLoggerCallback(meta_writer))

    return kwargs


# pylint: disable=unused-argument
def dirname_creator(trial: ray.tune.experiment.Trial) -> str:
    return trial.trial_id.split('_')[0]


class InfoToCustomMetricsCallback(
        ray.rllib.algorithms.callbacks.DefaultCallbacks):
    # pylint: disable=arguments-differ
    def on_episode_end(
        self,
        *_: Any,
        episode: Union[Episode, EpisodeV2, Exception],
        **__: Any,
    ) -> None:
        assert isinstance(episode, Episode)
        for info in episode._agent_to_last_info.values():
            episode.custom_metrics.update(
                ray.tune.utils.flatten_dict(info.copy()))


def gen_passwd(size: int) -> str:
    """Generate password for ray.init call.

    See
    https://docs.ray.io/en/latest/configure.html?highlight=security#redis-port-authentication

    This function was adapted from https://stackoverflow.com/a/2257449

    Example
    -------
    ray.init(redis_password=gen_passwd(512))

    Parameters
    ----------
    size : int
        how long the password should be

    """
    # https://stackoverflow.com/a/2257449
    chars = string.ascii_letters + string.digits
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))


def check_nan(x: torch.Tensor, *args: Any, **kwargs: Any) -> None:
    if not torch.any(torch.isnan(x)) and not torch.any(torch.isinf(x)):
        return

    msg = f"nan detected. x is {x}\n"
    if args:
        msg += "check_nan args are\n"
        for arg in args:
            msg += f"{arg}\n"

    if kwargs:
        msg += "check_nan kwargs are\n"
        for key, val in kwargs.items():
            msg += f"{key}: {val}\n"

    raise ValueError(msg)
