import sys
import collections
import functools
import glob
import re
import difflib
import importlib
import pickle
import copy
import shutil
import pprint
import subprocess as sp
from pathlib import Path
from typing import Iterable, List, Union, Dict, Tuple, Optional, Sequence, \
    Any, TypedDict, TypeVar

import numpy as np
import pandas as pd

import plotly
import plotly.graph_objects as go

import yaml


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
            symlink_dir: Optional[str] = "."):

        self.files = [Path(f).absolute() for f in files]
        self.str_data = str_data or {}
        self.objs_to_pickle = objs_to_pickle
        self.print_log_dir = print_log_dir
        self.symlink_dir = \
            Path(symlink_dir).expanduser() if symlink_dir else None

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


def merge_dfs(
    dfs: Sequence[pd.DataFrame],
    names: Optional[Sequence[str]] = None
) -> pd.DataFrame:

    df = dfs[0]
    if len(dfs) > 1:
        for i, _df in enumerate(dfs[1:]):
            df = df.join(_df, how='outer', rsuffix=f'_{i+1}')
        df.columns = names or [f'values_{i}' for i in range(len(dfs))]
    return df


def shorten_dfs(_dfs: Sequence[pd.DataFrame], max_step: int = None) -> None:
    if not _dfs:
        return

    if max_step is None:
        # shortest maximum step among the dfs
        max_step = min(_df.index.max() for _df in _dfs)
    for i, _df in enumerate(_dfs):
        _dfs[i] = _df[_df.index <= max_step]  # type: ignore


def group_experiments(base_dir: Path) -> Dict[str, List[Path]]:
    # https://stackoverflow.com/a/57594612
    progress_files = glob.glob(
        str(base_dir) + '/**/progress.csv', recursive=True)

    out: Dict[str, List[Path]] = collections.defaultdict(list)
    for f in progress_files:
        name = '_'.join(Path(f).parent.name.split('_')[:3])
        out[name].append(Path(f).parent)

    return {k: sorted(v) for k, v in out.items()}


def get_progress(
    experiments: Iterable[Path],
    x_tag: str = 'timesteps_total',
    tag: str = 'episode_reward_mean',
    only_complete_data: bool = False,
    max_x: Optional[Any] = None,
    names: Optional[Sequence[str]] = None
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:

    def _print_suggestions(_word: str, _possibilities: List[str]) -> None:
        _suggestions = difflib.get_close_matches(_word, _possibilities)

        if _suggestions:
            print(f'suggestions for "{_word}": {", ".join(_suggestions)}')

    dfs = []
    for exp in experiments:
        try:
            df = pd.read_csv(exp / 'progress.csv', low_memory=False)
        except pd.errors.EmptyDataError:
            print('{exp} has empty progress.csv, skipping')
            continue

        try:
            df = df[[x_tag, tag]].set_index(x_tag)
        except KeyError as e:
            keys = list(df.columns)
            print('Error setting index for')
            print(str(exp))
            print('available keys are')
            pprint.pprint(keys)
            if x_tag not in keys:
                _print_suggestions(x_tag, keys)
            if tag not in keys:
                _print_suggestions(tag, keys)
            print('skipping')
            raise e

        dfs.append(df)

    if only_complete_data:
        shorten_dfs(dfs)

    if max_x is not None:
        shorten_dfs(dfs, max_x)
    if not names:
        names = []
        for exp in experiments:
            try:
                names.append(exp.name.split('_')[3])
            except IndexError:
                names.append(exp.name)

    return merge_dfs(dfs, names), dfs


# pylint: disable=too-many-locals
def plot_progress(
    dfs: Dict[str, pd.DataFrame],
    plot_indiv: bool = True,
    indiv_alpha: float = 0.2,
    percentiles: Optional[Tuple[float, float]] = None,
    percentile_alpha: float = 0.1,
) -> go.Figure:
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS

    assert 0 <= indiv_alpha <= 1
    assert 0 <= percentile_alpha <= 1

    if percentiles is not None:
        assert len(percentiles) == 2 and \
            0 <= percentiles[0] <= 1 and \
            0 <= percentiles[1] <= 1, "percentiles must be between 0 and 1"

    def _make_transparency(_color: str, _alpha: float) -> str:
        return f'rgba({colors[i][4:-1]}, {_alpha})'

    buttons = []
    traces: List[go.Scatter] = []
    for i, (name, df) in enumerate(dfs.items()):

        df = df[~np.isnan(df.mean(axis=1))]
        beg_traces = len(traces)

        traces.append(go.Scatter(
            x=df.index, y=df.mean(axis=1), name=name, showlegend=True,
            line_color=colors[i], line_width=2, mode='lines',
            hoverlabel_namelength=-1
        ))

        if plot_indiv:
            clr = _make_transparency(colors[i], indiv_alpha)
            for col in df.columns:
                traces.append(go.Scatter(
                    x=df.index, y=df[col], name=col, showlegend=False,
                    line_color=clr, mode='lines',
                    hoverlabel_namelength=-1,
                    hoverinfo='none'
                ))

        if len(df.columns) > 1 and percentiles:
            fill_clr = _make_transparency(colors[i], percentile_alpha)
            line_clr = _make_transparency(colors[i], 0.0)

            plt = functools.partial(
                go.Scatter, x=df.index,
                showlegend=False, line_color=line_clr, mode='lines',
                hoverlabel_namelength=-1)
            traces.append(plt(y=df.quantile(percentiles[0], axis=1),
                              name=f'{name}-{round(100 * percentiles[0])}%'))
            traces.append(plt(y=df.quantile(percentiles[1], axis=1),
                          name=f'{name}-{round(100 * percentiles[1])}%',
                          fill='tonexty', fillcolor=fill_clr))

        idxs = list(range(beg_traces, len(traces)))
        buttons.append({
            'method': 'restyle',
            'label': name,
            'visible': True,
            'args': [{'visible': True}, idxs],
            'args2': [{'visible': 'legendonly'}, idxs],
        })

    buttons.append({
        'method': 'restyle',
        'label': 'all',
        'visible': True,
        'args': [{'visible': True}],
        'args2': [{'visible': 'legendonly'}]
    })

    # https://stackoverflow.com/a/65955881
    layout = go.Layout(
        updatemenus=[{
            'type': 'buttons',
            'direction': 'left',
            'x': 1,
            'y': -0.1,
            'showactive': True,
            'buttons': buttons
        }],
        showlegend=True,
        hovermode='x unified',
        hoverlabel_bgcolor='rgba(255, 255, 255, 0.5)')

    return go.Figure(data=traces, layout=layout)


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
