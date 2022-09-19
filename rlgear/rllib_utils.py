import argparse
import uuid
import os
import string
import random
from typing import Tuple, Any, Iterable, Union, Dict

import yaml
import ray
import ray.tune.utils

from .utils import MetaWriter, get_inputs, parse_inputs, get_log_dir, \
    StrOrPath, dict_str2num, import_class


class MetaLoggerCallback(ray.tune.logger.LoggerCallback):
    def __init__(self, meta_writer: MetaWriter):
        self.meta_writer = meta_writer

    def log_trial_start(self, trial: ray.tune.experiment.trial.Trial) -> None:
        self.meta_writer.write(trial.logdir)


def add_rlgear_args(parser: argparse.ArgumentParser) \
        -> argparse.ArgumentParser:
    parser.add_argument('yaml_file')
    parser.add_argument('exp_name')
    parser.add_argument('--overrides')
    parser.add_argument('--debug', action='store_true')
    return parser


def make_rllib_config(
    yaml_file: StrOrPath,
    exp_name: str,
    search_dirs: Union[StrOrPath, Iterable[StrOrPath]],
    debug: bool,
    overrides: dict,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    inputs = get_inputs(yaml_file, search_dirs)
    params = dict_str2num(parse_inputs(inputs))
    # override the non-rllib blocks
    if overrides is not None:
        params = ray.tune.utils.merge_dicts(params, overrides)

    # loggers = list(ray.tune.logger.DEFAULT_LOGGERS)
    meta_writer = MetaWriter(
        repo_roots=params['git_repos'],
        files=inputs,
        str_data={'merged_params.yaml': yaml.dump(params)})

    # provide defaults that can be overriden in the yaml file
    kwargs: dict = {
        'config': {
            "log_level": "INFO",
        },
        'local_dir': str(get_log_dir(params['log'], yaml_file, exp_name)),
        # 'loggers': loggers
    }

    for blk in params['rllib']['tune_kwargs_blocks'].split(','):
        kwargs = ray.tune.utils.merge_dicts(kwargs, params['rllib'][blk])

    if debug:
        kwargs['local_dir'] = os.path.join(kwargs['local_dir'], 'debug')
        kwargs['config']['num_workers'] = 0
        kwargs['config']['num_gpus'] = 0
        kwargs['num_samples'] = 1
        if kwargs['verbose'] == 0:
            kwargs['verbose'] = 1
        if kwargs['config']['log_level'] in ['ERROR', 'WARN']:
            kwargs['config']['log_level'] = 'INFO'

    # handle the rllib logger callbacks. These are more complicated because
    # they need to be classes, not the objects. For now just handling
    # the case of a single callback but if needed in the future add support
    # for callbacks.MultiCallbacks
    if 'callbacks' in kwargs['config']:
        kwargs['config']['callbacks'] = \
            import_class(kwargs['config']['callbacks'])

    # handle the tune logger callbacks
    if 'callbacks' not in kwargs:
        kwargs['callbacks'] = []
    else:
        for i, cb in enumerate(kwargs['callbacks']):
            if isinstance(cb, dict):
                kwargs['callbacks'][i] = import_class(kwargs['callbacks'][i])

    kwargs['callbacks'].append(MetaLoggerCallback(meta_writer))

    meta_writer.objs_to_pickle = kwargs  # type: ignore

    return params, kwargs


# pylint: disable=unused-argument
def dirname_creator(trial: ray.tune.experiment.Trial) -> str:
    return str(uuid.uuid4())[:5]


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
