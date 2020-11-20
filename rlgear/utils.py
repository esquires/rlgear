import sys
import importlib
import copy
import shutil
import re
import subprocess as sp
from pathlib import Path
from typing import Iterable, List, Union, Dict, Tuple, Optional, Sequence, \
    Any

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import yaml
import git

StrOrPath = Union[str, Path]
GymObsRewDoneInfo = Tuple[np.ndarray, float, bool, dict]


class MetaWriter():
    def __init__(
            self,
            repo_roots: Union[Iterable[StrOrPath], StrOrPath],
            files: Union[Iterable[StrOrPath], StrOrPath],
            str_data: Optional[Dict[str, str]] = None,
            print_log_dir: bool = True,
            symlink_dir: str = ".",
            check_clean: Union[Iterable[bool], bool] = False):

        def _to_list_of_paths(var: Union[Iterable[StrOrPath], StrOrPath]) \
                -> List[Path]:
            var_list = [var] if isinstance(repo_roots, (str, Path)) else var
            return [Path(v) for v in var_list]  # type: ignore

        roots = _to_list_of_paths(repo_roots)
        if not isinstance(check_clean, list):
            check_clean = [bool(check_clean) for _ in range(len(roots))]

        self.files = _to_list_of_paths(files)
        self.str_data = str_data or {}
        self.symlink_dir = Path(symlink_dir).expanduser()

        try:
            self.requirements = sp.check_output(
                ['pip3', 'freeze']).decode(encoding='UTF-8')
        except FileNotFoundError:
            self.requirements = sp.check_output(
                ['pip', 'freeze']).decode(encoding='UTF-8')

        self.print_log_dir = print_log_dir

        self.cmd = " ".join(sys.argv)

        self.git_info = {}
        for repo_root, check in zip(roots, check_clean):
            try:
                repo = git.Repo(repo_root,
                                search_parent_directories=True)
                diff = sp.check_output(
                    ['git', 'diff'],
                    cwd=repo.working_dir).decode(encoding='UTF-8')

            # pylint: disable=no-member
            except (git.exc.InvalidGitRepositoryError,
                    git.exc.NoSuchPathError,
                    sp.CalledProcessError) as e:
                print(e)
                print((f'ignoring the previous git error for {repo_root}.'
                       'This git repo will not be saved.'))
            else:
                if check:
                    assert not repo.head.commit.diff(None), \
                        (f"check_clean is set to True for {repo.common_dir} "
                         "but the status is not clean")
                self.git_info[Path(repo.working_dir).stem] = {
                    'repo_dir': repo.working_dir,
                    'commit': repo.commit().name_rev,
                    'diff': diff}

    def write(self, logdir: str) -> None:
        if self.print_log_dir:
            print(f'log dir: {logdir}')

        if self.symlink_dir:
            link_tgt = self.symlink_dir / 'latest'
            try:
                # can add missing_ok in python 3.8
                link_tgt.unlink()
            except FileNotFoundError:
                pass
            link_tgt.symlink_to(logdir, target_is_directory=True)

        meta_dir = Path(logdir) / 'meta'
        meta_dir.mkdir(exist_ok=True)

        for fname_str, data in self.str_data.items():
            with open(meta_dir / fname_str, 'w') as f:
                f.write(data)

        for fname in self.files:
            shutil.copy2(fname, meta_dir)

        with open(meta_dir / 'args.txt', 'w') as f:
            f.write(self.cmd)

        with open(meta_dir / 'requirements.txt', 'w') as f:
            f.write(self.requirements)

        for repo_name in self.git_info:
            commit = self.git_info[repo_name]['commit']
            commit_only = commit.split(" ")[0]
            repo_dir = self.git_info[repo_name]['repo_dir']

            meta_repo_dir = meta_dir / repo_name
            meta_repo_dir.mkdir(exist_ok=True)

            with open(meta_repo_dir / (repo_name + '_commit.txt'), 'w') as f:
                f.write(commit)

            code = [
                "import subprocess as sp",
                f"repo_dir = '{repo_dir}'",
                f"print('stashing changes in {repo_dir}')",
                "sp.call(['git', 'stash'], cwd=repo_dir)",
                f"sp.call(['git', 'checkout', '{commit_only}'], cwd=repo_dir)",
            ]

            if self.git_info[repo_name]['diff']:
                diff_file = meta_repo_dir / (repo_name + '_diff.diff')
                with open(diff_file, 'w') as f:
                    f.write(self.git_info[repo_name]['diff'])
                code.append(
                    (f"sp.call(['git', 'apply', '{diff_file.resolve()}'], "
                     "cwd=repo_dir)"))

            with open(meta_repo_dir / (repo_name + '_restore.py'), 'w') as f:
                f.write("\n".join(code))


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
        with open(filepath, 'r') as f:
            params = yaml.safe_load(f)

        temp_inputs = params.get('__inputs__', [])
        if isinstance(temp_inputs, str):
            temp_inputs = [temp_inputs]

        for inp in temp_inputs:
            _get_inputs(inp)

        inputs.append(filepath)

    _get_inputs(yaml_file)
    return inputs


def parse_inputs(inputs: Iterable[StrOrPath]) -> dict:
    # note that importing ray.tune does not override tf.executing_eagerly
    # pylint: disable=import-outside-toplevel
    from ray.tune.utils import merge_dicts

    out: dict = {}
    for inp in inputs:
        with open(inp, 'r') as f:
            params = yaml.safe_load(f)

        out = merge_dicts(out, params)

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


def dict_import_class(d: dict) -> dict:
    keys = list(d.keys())  # copy
    tgt_keys = {'cls', 'kwargs', "__dict_import_class"}
    for k in keys:
        if isinstance(d[k], dict):
            d[k] = dict_import_class(d[k])

            if set(d[k].keys()) == tgt_keys and d[k]["__dict_import_class"]:
                d[k] = import_class(d[k])

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
    if max_step is None:
        # shortest maximum step among the dfs
        max_step = min([_df.index.max() for _df in _dfs])
    for i, _df in enumerate(_dfs):
        _dfs[i] = _df[_df.index <= max_step]  # type: ignore


def preprocess_pbt_df(df: pd.DataFrame, x_tag: str) -> pd.DataFrame:
    """Align the x_tag index used in plot_progress before plotting.

    Args:
    ----
        df (pd.DataFrame): Dataframe of the progress.csv file
        x_tag (str): Desired tag in df to use as index of plot

    Return:
    ------
        df (pd.DataFrame): Dataframe with the x_tag column aligned to plot

    """
    tag_diff = '{}_diff'.format(x_tag)
    df[tag_diff] = df[x_tag].diff()
    # find the indexes where overlap starts
    neg_diff = df.loc[df[tag_diff] < 0].index.to_list()
    # if no overlap is found return the original dataframe
    if len(neg_diff) == 0:
        return df
    else:
        for val in neg_diff:
            start = val
            end = len(df)
            # shift x_tag values by the overlapping amount
            df[x_tag][start:end] += df[tag_diff][start-2] + \
                abs(df[tag_diff][start])
    return df


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


# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
def plot_progress(
        ax: plt.axis,
        base_dirs: Iterable[StrOrPath],
        tag: str,
        x_tag: str = 'timesteps_total',
        preprocess_pbt: bool = False,
        names: Optional[Iterable[str]] = None,
        only_complete_data: bool = False,
        show_same_num_timesteps: bool = False,
        percentiles: Optional[Tuple[float, float]] = None,
        alpha: float = 0.1,
        xtick_interval: Optional[float] = None,
        max_step: int = None) -> None:

    base_dirs = [d for d in base_dirs if Path(d).is_dir()]
    if not names:
        names = [Path(d).name for d in base_dirs]

    out_dfs = []
    for d in base_dirs:
        progress_files = list(Path(d).rglob('progress.csv'))

        # this is not a list comprehension because there are cases
        # where the progress.csv file exists but is empty
        dfs = []
        for fname in progress_files:
            try:
                dfs.append(pd.read_csv(fname))
            except pd.errors.EmptyDataError:
                print(f'{fname} is empty, skipping')

        for i, df in enumerate(dfs):
            try:
                if preprocess_pbt:
                    df = preprocess_pbt_df(df, x_tag)
                dfs[i] = df[[x_tag, tag]].set_index(x_tag)
            except KeyError as e:
                print(f'available keys are {df.columns}')
                raise e

        if only_complete_data:
            shorten_dfs(dfs)

        out_dfs.append(merge_dfs(dfs))

    if show_same_num_timesteps:
        shorten_dfs(out_dfs)

    if max_step is not None:
        shorten_dfs(out_dfs, max_step)

    for name, df in zip(names, out_dfs):
        ax.plot(df.index, df.mean(axis=1), label=name)

        if percentiles:
            plot_percentiles(ax, df, percentiles, alpha)

    if xtick_interval:
        ax.xaxis.set_major_locator(mtick.MultipleLocator(xtick_interval))


def import_class(class_info: Union[str, dict]) -> Any:
    def _get_class(class_str: str) -> Any:
        split = class_str.split('.')
        return getattr(
            importlib.import_module('.'.join(split[:-1])), split[-1])

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
