import copy
import csv
import numbers
import os
import random
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Union

import numpy as np

try:
    from torch import Tensor
except ImportError:
    Tensor = None  # type: ignore

import ray
import ray.rllib.algorithms.callbacks
import ray.tune.registry
import ray.tune.trainable.trainable
import ray.tune.utils
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.tune.experiment.trial import Trial
from ray.tune.registry import ENV_CREATOR, _global_registry

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = object  # type: ignore

from .utils import MetaWriter, StrOrPath, import_class


class MetaLoggerCallback(ray.tune.logger.LoggerCallback):
    """Wrap :class:`rlgear.utils.MetaWriter`.

    For use with :class:`ray.tune.logger.LoggerCallback`

    Example
    -------
    .. code-block:: python

        ray.tune.run(
            callbacks=[MetaLoggerCallback(MetaWriter())],
            ...
        )

    """

    def __init__(self, meta_writer: MetaWriter):
        self.meta_writer = meta_writer

    def log_trial_start(self, trial: ray.tune.experiment.trial.Trial) -> None:
        self.meta_writer.write(trial.logdir)


class Filter:
    """Filter extra :class:`ray.tune.logger.LoggerCallback` outputs.

    Parameters
    ----------
    excludes: List[str]
        list of regexes to be compiled via :func:`re.compile`

    """

    def __init__(self, excludes: List[str]):
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


class TBXFilteredLoggerCallback(
            ray.tune.logger.tensorboardx.TBXLoggerCallback):
    """Wrap :class:`ray.tune.logger.tensorboardx.TBXLoggerCallback`.

    Reduces the output based on the provided :func:`Filter`.
    """

    def __init__(self, filt: Filter, prefixes: dict[str, str]):
        super().__init__()
        self.filt = filt
        self.prefixes = prefixes
        self._summary_writer_cls = self._summary_writer_cls_rm_prefix

    def log_trial_result(
        self,
        iteration: int,
        trial: ray.tune.experiment.trial.Trial,
        result: Dict[str, Any]
    ) -> None:
        super().log_trial_result(iteration, trial, self.filt(result))

    def _summary_writer_cls_rm_prefix(self, *args: Any, **kwargs: Any) \
            -> SummaryWriterAdjPrefix:
        return SummaryWriterAdjPrefix(self.prefixes, *args, **kwargs)


class JsonFiltredLoggerCallback(ray.tune.logger.json.JsonLoggerCallback):
    """Wrap :class:`ray.tune.logger.json.JsonLoggerCallback`.

    Reduces the output based on the provided :func:`Filter`.
    """

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
    """Wrapper around :class:`ray.tune.logger.csv.CSVLoggerCallback` \
        that reduces the output based on the provided excludes regexes.

    This callback also waits a set number of training iterations before
    freezing the keys as sometimes not all logging items are available
    on the first iteration.
    """

    def __init__(self, wait_iterations: int, excludes: List[str]):
        super().__init__()
        self.wait_iterations = wait_iterations
        self.prior_results: dict[Trial, list[dict[str, Any]]] = defaultdict(list)
        self.keys: dict[Trial, set[str]] = defaultdict(set)
        self.filt = Filter(excludes)
        self.excluded_keys: set[str] = set()

    def log_trial_result(
        self, iteration: int, trial: Trial, result: dict[str, Any]
    ) -> None:

        # see piece of ray.tune.logger.csv.CSVLoggerCallback:log_trial_result
        if trial not in self._trial_files:
            self._setup_trial(trial)

        training_iteration = result['training_iteration']
        result = self.filt(result)
        self.excluded_keys.update({
            k for k, r in result.items() if not isinstance(r, numbers.Real)
        })
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


def get_trainer(tune_kwargs: dict[Any, Any]) -> ray.tune.trainable.trainable.Trainable:
    trainer_cls = ray.tune.registry.get_trainable_cls(tune_kwargs["run_or_experiment"])
    trainer = trainer_cls(config=tune_kwargs["config"])
    if tune_kwargs["restore"]:
        trainer.restore(str(Path(tune_kwargs["restore"]).expanduser()))
    return trainer


def make_env(env: str, cfg: dict[str, Any], run_check: bool) -> Any:
    env_creator = _global_registry.get(ENV_CREATOR, env)  # type: ignore
    env = env_creator(cfg)
    return env


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
            cfg['callbacks'] = ray.rllib.algorithms.callbacks.make_multi_callbacks(
                [import_class(cb) for cb in cfg['callbacks']])  # type: ignore

    # handle the tune logger callbacks
    if 'callbacks' not in kwargs:
        kwargs['callbacks'] = []
    else:
        for i, cb in enumerate(kwargs['callbacks']):
            if isinstance(cb, dict):
                kwargs['callbacks'][i] = import_class(kwargs['callbacks'][i])

    excludes = params['log'].get('callbacks', {}).get('excludes', [])

    if 'csv' in params['log']['callbacks']:
        kwargs['callbacks'].append(CSVFilteredLoggerCallback(
            params['log']['callbacks']['csv']['wait_iterations'], excludes))

    if 'tensorboard' in params['log']['callbacks']:
        kwargs['callbacks'].append(TBXFilteredLoggerCallback(
            Filter(excludes),
            params['log']['callbacks']['tensorboard'].get('prefixes', {})
        ))

    if 'json' in params['log']['callbacks']:
        kwargs['callbacks'].append(JsonFiltredLoggerCallback(Filter(excludes)))

    if 'tune_kwargs' not in meta_writer.str_data:
        kwargs_copy = copy.deepcopy(kwargs)
        try:
            del kwargs_copy['config']['callbacks']
        except KeyError:
            pass

        meta_writer.str_data['tune_kwargs.yaml'] = kwargs
        meta_writer.objs_to_pickle['tune_kwargs.p'] = kwargs_copy

    kwargs['callbacks'].append(MetaLoggerCallback(meta_writer))

    return kwargs


# pylint: disable=unused-argument
def dirname_creator(trial: ray.tune.experiment.Trial) -> str:
    return trial.trial_id.split('_')[0]


class InfoToCustomMetricsCallback(
        ray.rllib.algorithms.callbacks.DefaultCallbacks):
    # pylint: disable=arguments-differ,no-self-use
    def on_episode_end(
        self,
        *_: Any,
        episode: Union[Episode, EpisodeV2, Exception],
        **__: Any,
    ) -> None:

        if isinstance(episode, Episode):
            for info in episode._agent_to_last_info.values():
                episode.custom_metrics.update(
                    ray.tune.utils.flatten_dict(info.copy()))
        else:
            episode.custom_metrics.update(
                ray.tune.utils.flatten_dict(episode._last_infos)
            )


def gen_passwd(size: int) -> str:
    """Generate password for ray.init call.

    See
    https://docs.ray.io/en/latest/configure.html?highlight=security#redis-port-authentication

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
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))


def check(
    x: Tensor,
    *args: Any,
    lim: float = 1.0e7,
    **kwargs: Any
) -> None:
    """Raise :py:exc:`ValueError` if ``|x|`` has large or ``nan`` entries.

    Parameters
    ----------
    x : torch.Tensor
        a tensor to be checked for large or ``nan`` values
    args : Any
        these will be printed out in order when ``|x|`` is large or has \
        ``nan`` values
    lim : float, default 1.0e7
        value that defines whether ``|x|`` is large
    kwargs : Any
        these will be printed out in ``|x|`` is large or has ``nan`` values

    """
    import torch  # pylint: disable=import-outside-toplevel
    failed = torch.any(torch.isnan(x))
    isinf = torch.any(torch.isinf(x))
    if np.isinf(lim):
        failed |= isinf
    else:
        failed |= isinf or torch.any(torch.abs(x.float()) >= lim)

    if not failed:
        return

    msg = f"check failed with limit {lim}. x is {x}\n"
    if args:
        msg += "check args are\n"
        for arg in args:
            msg += f"{arg}\n"

    if kwargs:
        msg += "check kwargs are\n"
        for key, val in kwargs.items():
            msg += f"{key}: {val}\n"

    raise ValueError(msg)
