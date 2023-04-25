import os
import fnmatch
from pathlib import Path
from typing import Iterable, Any, Set, Tuple, Type

import numpy as np
import gym

import ray.tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .utils import get_inputs, dict_str2num, parse_inputs, StrOrPath


def make_dummy_env(
        ob_space: gym.Space, ac_space: gym.Space, multiagent: bool) \
        -> Any:
    # https://github.com/ray-project/ray/issues/6809
    class Empty:
        pass

    base = MultiAgentEnv if multiagent else Empty

    # pylint: disable=abstract-method
    class DummyEnv(base):  # type: ignore
        observation_space = ob_space
        action_space = ac_space

        # pylint: disable=super-init-not-called
        def __init__(self, env_config: dict):
            pass

    return DummyEnv


def make_agent(yaml_file: Path, search_dirs: Iterable[StrOrPath]) \
        -> Tuple[Type[Algorithm], dict]:
    inputs = get_inputs(yaml_file, search_dirs)
    params = dict_str2num(parse_inputs(inputs))

    kwargs: dict = {}
    for blk in params['rllib']['tune_kwargs_blocks'].split(','):
        kwargs = ray.tune.utils.merge_dicts(kwargs, params['rllib'][blk])

    trainer_cls = ray.tune.registry.get_trainable_cls(
        kwargs['run_or_experiment'])

    if 'callbacks' in kwargs['config']:
        del kwargs['config']['callbacks']

    # https://github.com/ray-project/ray/issues/6809
    kwargs['config']['num_gpus'] = 0
    kwargs['config']['num_workers'] = 0

    # none of these values is used at test time
    for k in ['gamma', 'lambda', 'entropy_coeff', 'lr', 'kl_coeff',
              'num_sgd_iter', 'rollout_fragment_length',
              'sgd_minibatch_size', 'train_batch_size']:
        kwargs['config'][k] = 0

    return trainer_cls, kwargs


def ckpt_to_yaml(ckpt: Path) -> Path:
    d = ckpt.parent
    out = d / 'meta' / 'merged_params.yaml'
    root = Path('/')
    while not out.exists() and d != root:
        d = d.parent
        out = d / 'meta' / 'merged_params.yaml'

    if not out.exists():
        raise FileNotFoundError(f'could not find yaml file for {ckpt}')

    return out


class SelfPlay:
    def __init__(self, prev_paths: Iterable[str], watch_path: str):
        self.watch_path = watch_path
        print(('SelfPlay: building initial checkpoints set. '
               'This may take a while...'))
        prior_ckpts_set: Set[Path] = set()
        for p in prev_paths:
            self._build_ckpts(p, prior_ckpts_set)
        self.prior_ckpts = list(prior_ckpts_set)
        print('done building initial checkpoints set')

    # pylint: disable=no-self-use
    def _build_ckpts(self, path: str, ckpts: Set[Path]) \
            -> None:
        # Path.rglob gave errors when there are temporary files created
        # during the glob
        for root, _, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, '*.tune_metadata'):
                ckpt = (Path(root) / filename).with_suffix('')
                if ckpt.exists():
                    ckpts.add(ckpt)
                else:
                    print((f"checkpoint {ckpt} created after metadata. "
                           "This is not typical..."))

    def get_ckpt(self) -> Path:
        watched_ckpts_set: Set[Path] = set()
        self._build_ckpts(self.watch_path, watched_ckpts_set)
        watched_ckpts = list(watched_ckpts_set)

        while True:
            if len(self.prior_ckpts) + len(watched_ckpts) == 0:
                raise ValueError('no checkpoints available')

            idx = int(np.random.randint(
                len(self.prior_ckpts) + len(watched_ckpts)))
            if idx < len(self.prior_ckpts):
                ckpt = self.prior_ckpts[idx]
                if not ckpt.exists():
                    self.prior_ckpts.remove(ckpt)
                    continue
            else:
                ckpt = watched_ckpts[idx - len(self.prior_ckpts)]

            break

        return ckpt
