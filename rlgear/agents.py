import queue
import re
import os
import fnmatch
from pathlib import Path
from typing import Iterable, Any

import numpy as np
import gym
import boltons.setutils

import ray.tune
from ray.rllib.agents.trainer import Trainer

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

from .utils import get_inputs, dict_str2num, parse_inputs, StrOrPath


def make_dummy_env(ob_space: gym.Space, ac_space: gym.Space) \
        -> Any:
    # https://github.com/ray-project/ray/issues/6809
    class DummyEnv:
        observation_space = ob_space
        action_space = ac_space

        def __init__(self, env_config: dict):
            pass

    return DummyEnv


class CheckpointMonitor(FileSystemEventHandler):
    def __init__(self, created_queue: queue.Queue, deleted_queue: queue.Queue):
        self.created_queue = created_queue
        self.deleted_queue = deleted_queue
        super().__init__()

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.src_path.endswith('tune_metadata'):
            return

        ckpt = Path(event.src_path).with_suffix('')
        if ckpt.exists():
            self.created_queue.put(ckpt.resolve())
        else:
            print((f"checkpoint {ckpt} created after metadata. "
                   "This is not typical..."))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return

        ckpt = Path(event.src_path)
        m = re.match(r'checkpoint-\d+$', ckpt.name)
        if m is None:
            return

        self.deleted_queue.put(ckpt)


def make_agent(yaml_file: Path, search_dirs: Iterable[StrOrPath], env: Any) \
        -> Trainer:
    inputs = get_inputs(yaml_file, search_dirs)
    params = dict_str2num(parse_inputs(inputs))

    kwargs: dict = {}
    for blk in params['rllib']['tune_kwargs_blocks'].split(','):
        kwargs = ray.tune.utils.merge_dicts(kwargs, params['rllib'][blk])

    trainer_cls = ray.tune.registry.get_trainable_cls(
        kwargs['run_or_experiment'])

    # https://github.com/ray-project/ray/issues/6809
    kwargs['config']['num_gpus'] = 0
    kwargs['config']['num_workers'] = 0

    # none of these values is used at test time
    for k in ['gamma', 'lambda', 'entropy_coeff', 'lr', 'kl_coeff',
              'num_sgd_iter', 'rollout_fragment_length',
              'sgd_minibatch_size', 'train_batch_size']:
        kwargs['config'][k] = 0

    return trainer_cls(kwargs['config'], env=env)


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
        self.created_queue: queue.Queue = queue.Queue()
        self.deleted_queue: queue.Queue = queue.Queue()
        event_handler = CheckpointMonitor(
            self.created_queue, self.deleted_queue)
        self.observers = []

        self.observers.append(Observer())
        self.observers[-1].schedule(event_handler, watch_path, recursive=True)
        self.observers[-1].start()

        # Path.rglob gave errors when there is are temporary files created
        # during the glob
        print(('SelfPlay: building initial checkpoints set. '
               'This may take a while...'))
        self.ckpts = boltons.setutils.IndexedSet()
        for p in prev_paths:
            for root, _, filenames in os.walk(p):
                for filename in fnmatch.filter(filenames, '*.tune_metadata'):
                    ckpt = (Path(root) / filename).with_suffix('')
                    if ckpt.exists():
                        self.ckpts.add(ckpt)
                    else:
                        print((f"checkpoint {ckpt} created after metadata. "
                               "This is not typical..."))
        print('done building initial checkpoints set')

    def refresh_data(self) -> None:
        while True:
            try:
                ckpt = self.created_queue.get(timeout=0.01)
            except queue.Empty:
                break
            self.ckpts.add(ckpt)

        while True:
            try:
                ckpt = self.deleted_queue.get(timeout=0.01)
            except queue.Empty:
                break
            self.ckpts.remove(ckpt)

    def get_ckpt(self) -> Path:
        idx = np.random.randint(len(self.ckpts))
        return self.ckpts[idx]
