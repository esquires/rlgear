import sys
import importlib
import pickle
import copy
import shutil
import re
import pprint
import subprocess as sp
from pathlib import Path
from typing import Iterable, List, Union, Dict, Tuple, Optional, Sequence, \
    Any, TypedDict, TypeVar

import numpy as np
import pandas as pd

import yaml

import tqdm

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
except ImportError:
    import warnings
    warnings.warn(
        'matplotlib not found, not defining plot_percentiles or plot_progress')

else:
    def plot_percentiles(
            ax: plt.axis, df: pd.DataFrame, percentiles: Tuple[float, float],
            alpha: float) -> None:

        assert 0 <= alpha <= 1

        assert len(percentiles) == 2 and \
            0 <= percentiles[0] <= 1 and \
            0 <= percentiles[1] <= 1, "percentiles must be between 0 and 1"

        pct_low = df.quantile(percentiles[0], axis=1)
        pct_high = df.quantile(percentiles[1], axis=1)
        ax.fill_between(df.index, pct_low, pct_high, alpha=alpha)

    # pylint: disable=too-many-arguments
    def plot_progress(
            ax: plt.axis,
            dfs: List[pd.DataFrame],
            names: Iterable[str],
            percentiles: Optional[Tuple[float, float]] = None,
            alpha: float = 0.1,
            xtick_interval: Optional[float] = None) -> None:

        for name, df in zip(names, dfs):
            df = df[~np.isnan(df.mean(axis=1))]
            ax.plot(df.index, df.mean(axis=1), label=name)

            if percentiles:
                plot_percentiles(ax, df, percentiles, alpha)

        if xtick_interval:
            ax.xaxis.set_major_locator(mtick.MultipleLocator(xtick_interval))


StrOrPath = Union[str, Path]
GymObsRewDoneInfo = Tuple[np.ndarray, float, bool, dict]


# pylint: disable=too-many-instance-attributes,too-many-arguments
class MetaWriter():

    # pylint: disable=too-many-locals
    def __init__(
            self,
            repo_roots: Dict[str, Dict[str, Any]],
            files: Iterable[StrOrPath],
            str_data: Optional[Dict[str, str]] = None,
            objs_to_pickle: Optional[List[Any]] = None,
            print_log_dir: bool = True,
            symlink_dir: str = "."):

        self.files = files
        self.str_data = str_data or {}
        self.objs_to_pickle = objs_to_pickle
        self.print_log_dir = print_log_dir
        self.symlink_dir = Path(symlink_dir).expanduser()

        # https://stackoverflow.com/a/58013217
        self.requirements = sp.check_output(
            [sys.executable, '-m', 'pip', 'freeze']).decode('UTF-8')

        self.cmd = " ".join(sys.argv)

        self.git_info = {}
        try:
            # pylint: disable=import-outside-toplevel
            import git
        except ImportError:
            pass
        else:
            # avoids mypy error:
            # Trying to read deleted variable "exc"
            git_exc: Any = git.exc  # type: ignore

            for repo_root, repo_config in repo_roots.items():

                try:
                    repo = git.Repo(repo_root, search_parent_directories=True)
                    cwd = repo.working_dir
                    assert cwd is not None

                    def _get(_cmd: Iterable[str]) -> str:
                        # pylint: disable=cell-var-from-loop
                        return sp.check_output(
                            _cmd, cwd=cwd).decode('UTF-8')  # type: ignore

                    diff = _get(['git', 'diff'])
                    base_commit = repo_config['base_commit']
                    patch = _get(
                        ['git', 'format-patch', '--stdout', base_commit])
                    base_commit_hash = _get(
                        ['git', 'rev-parse', '--short', base_commit]).strip()

                # pylint: disable=no-member
                except (git_exc.InvalidGitRepositoryError,
                        git_exc.NoSuchPathError,
                        sp.CalledProcessError) as e:
                    print(e)
                    print((f'ignoring the previous git error for {repo_root}.'
                           'This git repo will not be saved.'))
                else:
                    if repo_config['check_clean']:
                        assert not repo.head.commit.diff(None), \
                            ("check_clean is set to True for "
                             f"{repo.common_dir} but the status is not clean")

                    self.git_info[Path(cwd).stem] = {
                        'commit': repo.commit().name_rev,
                        'patch': patch,
                        'patch_fname':
                            f'patch_rel_to_{base_commit_hash}.patch',
                        'diff': diff
                    }

    # pylint: disable=too-many-locals
    def write(self, logdir: str) -> None:
        if self.print_log_dir:
            print(f'log dir: {logdir}')

        if self.symlink_dir:
            link_tgt = self.symlink_dir / 'latest'
            link_tgt.unlink(missing_ok=True)

            try:
                link_tgt.symlink_to(logdir, target_is_directory=True)
            except FileExistsError:
                pass

        meta_dir = Path(logdir) / 'meta'
        meta_dir.mkdir(exist_ok=True)

        for fname_str, data in self.str_data.items():
            with open(meta_dir / fname_str, 'w', encoding='UTF-8') as f:
                f.write(data)

        if self.objs_to_pickle:
            with open(meta_dir / 'objs_to_pickle.p', 'wb') as fp:
                pickle.dump(self.objs_to_pickle, fp)

        for fname in self.files:
            shutil.copy2(fname, meta_dir)

        with open(meta_dir / 'args.txt', 'w', encoding='UTF-8') as f:
            f.write(self.cmd)

        with open(meta_dir / 'requirements.txt', 'w', encoding='UTF-8') as f:
            f.write(self.requirements)

        for repo_name, repo_data in self.git_info.items():
            meta_repo_dir = meta_dir / repo_name
            meta_repo_dir.mkdir(exist_ok=True)

            with open(meta_repo_dir / (repo_name + '_commit.txt'),
                      'w', encoding='UTF-8') as f:
                f.write(repo_data['commit'])

            diff_file = meta_repo_dir / (repo_name + '_diff.diff')
            with open(diff_file, 'w', encoding='UTF-8') as f:
                f.write(repo_data['diff'])

            patch_file = meta_repo_dir / repo_data['patch_fname']
            with open(patch_file, 'w', encoding='UTF-8') as f:
                f.write(repo_data['patch'])


def find_filepath(
        fname: StrOrPath,
        search_dirs: Union[StrOrPath, Iterable[StrOrPath]]) \
        -> Path:
    fname_path = Path(fname)
    if fname_path.exists():
        # absolute path or relative
        return fname_path
    else:
        # relative to search_dirs
        if isinstance(search_dirs, (str, Path)):
            search_dirs = [search_dirs]

        try:
            paths = next(
                list(Path(d).rglob(fname_path.name))
                for d in search_dirs)  # type: ignore
            if not paths:  # pylint: disable=no-else-raise
                raise FileNotFoundError(
                    f'could not find {fname} in {search_dirs}')
            else:
                return paths[0]
        except StopIteration as e:
            print(f'could not find {fname} in {search_dirs}')
            raise e


def get_inputs(
        yaml_file: StrOrPath,
        search_dirs: Union[StrOrPath, Iterable[StrOrPath]]) \
        -> List[Path]:

    inputs = []

    def _get_inputs(fname: StrOrPath) -> None:
        filepath = find_filepath(fname, search_dirs)
        with open(filepath, 'r', encoding='UTF-8') as f:
            params = yaml.safe_load(f)

        temp_inputs = params.get('__inputs__', [])
        if isinstance(temp_inputs, str):
            temp_inputs = [temp_inputs]

        for inp in temp_inputs:
            _get_inputs(inp)

        inputs.append(filepath)

    _get_inputs(yaml_file)
    return inputs


def parse_inputs(inputs: Sequence[StrOrPath]) -> dict:

    out: dict = {}
    for inp in inputs:
        with open(inp, 'r', encoding='UTF-8') as f:
            params = yaml.safe_load(f)

        if out:
            # pylint: disable=import-outside-toplevel
            from ray.tune.utils import merge_dicts
            out = merge_dicts(out, params)
        else:
            out = params

    try:
        del out['__inputs__']
    except KeyError:
        pass
    return out


def dict_str2num(d: dict) -> dict:

    keys = list(d.keys())  # copy
    for k in keys:
        v = d[k]
        try:
            fv = float(v)
            d[k] = int(fv) if fv.is_integer() else float(v)
        except (ValueError, TypeError):
            if isinstance(v, dict):
                d[k] = dict_str2num(v)

    return d


def get_latest_checkpoint(ckpt_root_dir: str) -> str:
    ckpts = [str(c) for c in Path(ckpt_root_dir).rglob('*checkpoint-*')
             if 'meta' not in str(c)]
    r = re.compile(r'checkpoint_(\d+)')
    ckpt_nums = [int(r.search(c).group(1)) for c in ckpts]  # type: ignore
    return ckpts[np.argmax(ckpt_nums)]


def get_log_dir(log_params: Dict[str, str],
                yaml_fname: StrOrPath,
                exp_name: str) \
        -> Path:
    prefix = next((Path(d).expanduser() for d in log_params['prefixes']
                   if Path(d).expanduser().is_dir()))
    return prefix / log_params['exp_group'] / Path(yaml_fname).stem / exp_name


def merge_dfs(_dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
    _df = _dfs[0]
    if len(_dfs) > 1:
        for i, __df in enumerate(_dfs[1:]):
            _df = _df.join(__df, how='outer', rsuffix=f'_{i+1}')
        cols = [f'values_{i}' for i in range(len(_dfs))]
        _df.columns = cols
    return _df


def shorten_dfs(_dfs: Sequence[pd.DataFrame], max_step: int = None) -> None:
    if not _dfs:
        return

    if max_step is None:
        # shortest maximum step among the dfs
        max_step = min(_df.index.max() for _df in _dfs)
    for i, _df in enumerate(_dfs):
        _dfs[i] = _df[_df.index <= max_step]  # type: ignore


# pylint: disable=too-many-locals,too-many-branches,too-many-arguments
def get_progress(
        base_dirs: Iterable[StrOrPath],
        tag: str,
        x_tag: str = 'timesteps_total',
        only_complete_data: bool = False,
        show_same_num_timesteps: bool = False,
        max_step: int = None,
        show_progress: bool = False) -> List[pd.DataFrame]:

    dirs = [d for d in base_dirs if Path(d).is_dir()]
    if dirs != list(base_dirs):
        print(('The following base_dirs were not found: '
               f'{set(base_dirs) - set(dirs)}'))

    out_dfs: List[pd.DataFrame] = []
    for d in tqdm.tqdm(dirs, desc="Reading files") if show_progress else dirs:
        progress_files = list(Path(d).rglob('progress.csv'))

        # this is not a list comprehension because there are cases
        # where the progress.csv file exists but is empty
        dfs = []
        for fname in progress_files:
            try:
                dfs.append(pd.read_csv(fname, low_memory=False))
            except pd.errors.EmptyDataError:
                print(f'{fname} is empty, skipping')

        i = 0
        while i < len(dfs):
            try:
                dfs[i] = dfs[i][[x_tag, tag]].set_index(x_tag)
                i += 1
            except KeyError as e:
                print(e)
                print(f'Error setting index for {progress_files[i]}')
                pprint.pprint(f'available keys are {list(dfs[i].columns)}')
                pprint.pprint('skipping')
                del dfs[i]

        if only_complete_data:
            shorten_dfs(dfs)

        if dfs:
            out_dfs.append(merge_dfs(dfs))

    if show_same_num_timesteps:
        shorten_dfs(out_dfs)

    if max_step is not None:
        shorten_dfs(out_dfs, max_step)

    return out_dfs


class ImportClassDict(TypedDict):
    cls: str
    kwargs: Dict[str, Any]


def import_class(class_info: Union[str, ImportClassDict]) -> Any:
    def _get_class(class_str: str) -> Any:
        _split = class_str.split('.')
        try:
            _module = importlib.import_module('.'.join(_split[:-1]))
            return getattr(_module, _split[-1])
        except ModuleNotFoundError:
            # e.g. when an object contains another object (e.g. a staticmethod
            # within a class)
            _module = importlib.import_module('.'.join(_split[:-2]))
            return getattr(getattr(_module, _split[-2]), _split[-1])

    if isinstance(class_info, str):
        return _get_class(class_info)
    else:
        kwargs = class_info['kwargs']
        keys = copy.deepcopy(list(kwargs.keys()))
        for key in keys:
            if key.startswith('__preprocess_'):
                kwargs[key.replace('__preprocess_', '')] = \
                    import_class(kwargs[key])
                del kwargs[key]

        try:
            return _get_class(class_info['cls'])(**class_info['kwargs'])
        except Exception as e:
            print(f'could not initialize {class_info}')
            raise e


def smooth(values: Sequence[float], weight: float) -> Sequence[float]:
    # https://stackoverflow.com/a/49357445
    smoothed = []
    smoothed.append(values[0])
    for v in values[1:]:
        if np.isnan(v):
            smoothed.append(smoothed[-1])
        else:
            smoothed.append(smoothed[-1] * weight + v * (1 - weight))

    return smoothed


def add_to_dict(overrides: dict, keys: List[str], val: Any) -> None:
    d = overrides
    for key in keys[:-1]:
        try:
            d = d[key]
        except KeyError:
            d[key] = {}
            d = d[key]

    d[keys[-1]] = val


T = TypeVar('T')


def interp(x: T, x_low: Any, x_high: Any, y_low: Any, y_high: Any) -> T:
    if x_low == x_high:
        return y_low
    else:
        pct = (x - x_low) / (x_high - x_low)
        return y_low + pct * (y_high - y_low)
