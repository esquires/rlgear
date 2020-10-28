import argparse
import re
import os
import sys
import string
import json
import random
import time
from pathlib import Path
from typing import Tuple, Any, Iterable, Union, Dict, List

import yaml
import ray
import ray.tune.utils
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.agents.callbacks import DefaultCallbacks

from .utils import MetaWriter, get_inputs, parse_inputs, get_log_dir, \
    StrOrPath, dict_str2num, import_class


def make_rllib_metadata_logger(meta_data_writer: MetaWriter) \
        -> ray.tune.logger.Logger:
    class RllibLogMetaData(ray.tune.logger.Logger):
        def _init(self) -> None:
            self.meta_data_writer = meta_data_writer
            meta_data_writer.write(self.logdir)
            self.saved_continue_script = False

        def on_result(self, _: dict) -> None:
            if self.saved_continue_script:
                return

            self.saved_continue_script = True
            # we log everything in the _init function but need to override
            # on_result since it is a virtual method
            log_path = Path(self.logdir).parent
            json_fnames = log_path.glob('*.json')
            json_fname = None
            for fname in json_fnames:
                with open(fname, 'r') as f:
                    d = json.load(f)
                if len(d['checkpoints']) == 1:
                    # not supporting more than 1 checkpoint
                    ckpt = d['checkpoints'][0]
                    if ckpt['logdir'] == self.logdir:
                        json_fname = fname
                        break

            if json_fname:
                with open(log_path / 'continue.py', 'w') as f:
                    f.write("\n".join([
                        "import subprocess as sp",
                        "from pathlib import Path",
                        "restore_files = (Path(__file__).resolve() / 'meta').rglob('*_restore.py')",  # NOQA, pylint: disable=line-too-long
                        "for f in restore_files:",
                        "    sp.call(['python', str(f)])",
                        f"cmd = ['{sys.executable}'] + {sys.argv}",
                        r'cmd += ["--overrides", "{\"rllib\": {\"resume\": \"LOCAL\"}}"]',  # NOQA, pylint: disable=line-too-long
                        "# make this checkpoint the most recent so it is",
                        "# picked up by the rllib resume API",
                        f"Path('{json_fname}').touch()",
                        f"sp.call(cmd, cwd='{os.getcwd()}')"]))

    return RllibLogMetaData


def make_filtered_tblogger(
        regex_filters: List[str],
        time_elapsed: float = 0,
        train_iters: int = 0) -> ray.tune.logger.Logger:
    class FilteredTbLogger(ray.tune.logger.TBXLogger):
        def _init(self) -> None:
            super()._init()
            self.last_time = time.perf_counter() - 1 - time_elapsed
            self.last_train_iter = -1 - train_iters

        def on_result(self, result: dict) -> None:
            t = time.perf_counter()
            training_iter = result['training_iteration']
            if (t - self.last_time < time_elapsed or
                    training_iter - self.last_train_iter < train_iters):
                return

            self.last_time = t
            self.last_train_iter = training_iter
            result['custom_metrics'] = \
                {k: v for k, v in result['custom_metrics'].items()
                 if not any([re.search(r, k) for r in regex_filters])}
            super().on_result(result)

    return FilteredTbLogger


def add_rlgear_args(parser: argparse.ArgumentParser) \
        -> argparse.ArgumentParser:
    parser.add_argument('yaml_file')
    parser.add_argument('exp_name')
    parser.add_argument('--overrides')
    return parser


def make_basic_rllib_config(
        yaml_file: StrOrPath,
        exp_name: str,
        search_dirs: Union[StrOrPath, Iterable[StrOrPath]],
        overrides: dict = None) \
        -> Tuple[Dict[str, Any], Dict[str, Any]]:

    inputs = get_inputs(yaml_file, search_dirs)
    params = dict_str2num(parse_inputs(inputs))

    # override the non-rllib blocks
    if overrides is not None:
        params = ray.tune.utils.merge_dicts(params, overrides)

    log_dir = get_log_dir(params['log'], yaml_file, exp_name)

    loggers = list(ray.tune.logger.DEFAULT_LOGGERS)

    meta_data_writer = MetaWriter(
        repo_roots=[Path.cwd()] + params['git_repos']['paths'],
        files=inputs,
        str_data={'merged_params.yaml': yaml.dump(params)},
        check_clean=params['git_repos']['check_clean'])
    loggers.append(make_rllib_metadata_logger(meta_data_writer))

    if 'tb_filter' in params['log']:
        loggers = \
            [l for l in loggers if l is not ray.tune.logger.TBXLogger]  # NOQA
        loggers.append(make_filtered_tblogger(**params['log']['tb_filter']))

    # provide defaults that can be overriden in the yaml file
    kwargs: dict = {
        'config': {
            "log_level": "INFO",
        },
        'local_dir': str(log_dir),
        'loggers': loggers
    }

    for blk in params['rllib']['tune_kwargs_blocks'].split(','):
        kwargs = ray.tune.utils.merge_dicts(kwargs, params['rllib'][blk])

    if 'callbacks' not in kwargs['config'] \
            or not kwargs['config']['callbacks']:
        kwargs['config']['callbacks'] = \
            ['rlgear.rllib_utils.InfoToCustomMetricsCallback']

    cb_classes = \
        [import_class(cls_str) for cls_str in kwargs['config']['callbacks']]
    kwargs['config']['callbacks'] = \
        make_callbacks(cb_classes) if len(cb_classes) > 1 else cb_classes[0]

    return params, kwargs


class InfoToCustomMetricsCallback(DefaultCallbacks):
    # pylint: disable=arguments-differ,no-self-use
    def on_episode_end(  # type: ignore
            self, *_,
            episode: MultiAgentEpisode,
            **__) -> None:
        key = list(episode._agent_to_last_info.keys())[0]
        ep_info = episode.last_info_for(key).copy()
        episode.custom_metrics.update(ray.tune.utils.flatten_dict(ep_info))


def make_callbacks(callback_classes: Iterable[DefaultCallbacks]) \
        -> Any:
    class ListOfCallbacks(DefaultCallbacks):
        def __init__(self, *args, **kwargs):  # type: ignore
            super().__init__(*args, **kwargs)

            self.callbacks = \
                [cb_cls(*args, **kwargs) for cb_cls in callback_classes]

        def on_episode_start(self, *args, **kwargs):  # type: ignore
            for cb in self.callbacks:
                cb.on_episode_start(*args, **kwargs)

        def on_episode_step(self, *args, **kwargs):  # type: ignore
            for cb in self.callbacks:
                cb.on_episode_step(*args, **kwargs)

        def on_episode_end(self, *args, **kwargs):  # type: ignore
            for cb in self.callbacks:
                cb.on_episode_end(*args, **kwargs)

        def on_postprocess_trajectory(self, *args, **kwargs):  # type: ignore
            for cb in self.callbacks:
                cb.on_postprocess_trajectory(*args, **kwargs)

        def on_sample_end(self, *args, **kwargs):  # type: ignore
            for cb in self.callbacks:
                cb.on_sample_end(*args, **kwargs)

        def on_train_result(self, *args, **kwargs):  # type: ignore
            for cb in self.callbacks:
                cb.on_train_result(*args, **kwargs)

    return ListOfCallbacks


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
