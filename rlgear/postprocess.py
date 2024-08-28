import collections
import difflib
import pprint
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional, Sequence, \
    Any, Callable

import numpy as np
import pandas as pd

try:
    import plotly
    import plotly.graph_objects as go
except (ModuleNotFoundError, ImportError):
    plotly = None

    class go:  # type: ignore
        Figure = None

from .utils import get_files


class ProgressReader:
    def __init__(self) -> None:
        self.df_cache: Dict[Path, pd.DataFrame] = {}
        self.empty_cache: set[Path] = set()

    # pylint: disable=too-many-locals
    def get_progress(
        self,
        experiments: Iterable[Path],
        x_tag: str = 'timesteps_total',
        tag: str = 'episode_reward_mean',
        only_complete_data: bool = False,
        max_x: Optional[Any] = None,
        names: Optional[Sequence[str]] = None,
        tag_cb: Optional[Callable[[Path, str, List[str]], str]] = None,
        df_cb: Optional[Callable[[Path, pd.DataFrame], pd.DataFrame]] = None,
        log_fname: str = "progress.csv",
    ) -> Tuple[Optional[pd.DataFrame], List[pd.DataFrame]]:
        """Convert list of directories with progress.csv into single dataframe.

        This function reads the progress.csv file in each directory and puts the
        data in the tag column into a dataframe column where x_tag is the index.

        Parameters
        ----------
        experiments : Iterable[Path]
            list of experiments that contain a progress.csv file
        x_tag : str (default 'timesteps_total')
            what column in progress.csv to use as the index of the dataframe
        tag : str (default 'episode_reward_mean')
            what column in progress.csv to use as the data of the dataframe
        only_complete_data : bool (default True)
            when different experiments are further along than others, don't include
            this extra data. This can be useful when averaging or computing
            percentiles (see :func:`plot_progress`)
        max_x : Optional[Any] (default None)
            limit data so that data beyond x_tag is not included. This can be
            useful when comparing two different sets of experiments where one
            lasted far longer than the other.
        names : Optional[Sequence[str]] (default None)
            names to be given to the columns in the dataframe. If not provided,
            this will be the sample number of the particular experiment. For
            example, if individual experiments are called
            ``SAC_DM-v0_a_00000_0`` and ``SAC_DM-v0_a_00001_0`` then
            names will by default be ``["00000", "00001"]``

        """
        def _print_suggestions(_word: str, _possibilities: List[str]) -> None:
            _suggestions = difflib.get_close_matches(_word, _possibilities, n=5)

            if _suggestions:
                print(f'suggestions for "{_word}": ')
                for _suggestion in _suggestions:
                    print(_suggestion)

        def _merge_dfs(
            _dfs: Sequence[pd.DataFrame],
            _names: Optional[Sequence[str]] = None
        ) -> pd.DataFrame:

            _df = _dfs[0]
            if len(_dfs) > 1:
                for i, __df in enumerate(_dfs[1:]):
                    _df = _df.join(__df, how='outer', rsuffix=f'_{i+1}')
                _df.columns = _names or [f'values_{i}' for i in range(len(_dfs))]
            return _df

        def _shorten_dfs(_dfs: Sequence[pd.DataFrame]) -> None:
            if not _dfs:
                return

            # shortest maximum step among the dfs
            _max_x = min(_df.index.max() for _df in _dfs)

            if max_x is not None:
                _max_x = min(max_x, _max_x)

            for i, _df in enumerate(_dfs):
                _dfs[i] = _df[_df.index <= _max_x]  # type: ignore

        dfs = []
        filtered_names = []
        for i, exp in enumerate(experiments):

            fname = exp / log_fname
            if fname in self.df_cache:
                df = self.df_cache[fname]
            else:
                try:
                    df = pd.read_csv(fname, low_memory=False, sep="\t")
                except pd.errors.EmptyDataError:
                    if exp not in self.empty_cache:
                        print(f'{exp} has empty {log_fname}, skipping')
                        self.empty_cache.add(exp)
                    continue
                self.df_cache[fname] = df

            avail_tags = list(df.columns)

            if tag_cb is not None:
                temp_x_tag = tag_cb(exp, x_tag, avail_tags)
                temp_tag = tag_cb(exp, tag, avail_tags)
            else:
                temp_x_tag = x_tag
                temp_tag = tag

            if df_cb is not None:
                df = df_cb(exp, df)

            try:
                df = df[[temp_x_tag, temp_tag]].set_index(temp_x_tag)
            except KeyError:

                print('-----------')
                print(f'Error setting index "{temp_x_tag}" and data col "{temp_tag}"')
                if temp_x_tag not in avail_tags:
                    print(f"{temp_x_tag} not in available tags")
                    _print_suggestions(temp_x_tag, avail_tags)
                if temp_tag not in avail_tags:
                    print(f"{temp_x_tag} not in available tags")
                    _print_suggestions(temp_tag, avail_tags)
                print(str(exp))
            else:
                filtered_names.append(names[i] if names else exp.name)
                dfs.append(df)

        if only_complete_data:
            _shorten_dfs(dfs)

        if max_x is not None:
            _shorten_dfs(dfs)

        return _merge_dfs(dfs, filtered_names) if dfs else None, dfs


def get_dataframes(
    experiments: Dict[str, Iterable[Path]],
    progress_reader: ProgressReader,
    smooth_weight: Optional[float] = None,
    **get_progress_kwargs: Any,
) -> Dict[str, pd.DataFrame]:
    """Wrapper for :func:`get_progress`

    Parameters
    ----------
    experiments : dict[str, Iterable[Path]]
        list of experiments that contain a progress.csv file. see
        :func:`group_experiments`
    smooth_weight: Optional[float]
        weight for call to :func:`smooth`
    get_progress_kwargs : Any
        configurations for :func:`get_progress`

    Returns
    -------
    dataframes : dict[str, pd.DataFrame]
        keys match those in experiments

    """
    dfs = {}

    for nm, exp in experiments.items():

        df = progress_reader.get_progress(exp, **get_progress_kwargs)[0]

        if df is not None:
            if smooth_weight is not None:
                for col in df.columns:
                    df[col] = smooth(df[col].values, smooth_weight)

            dfs[nm] = df

    return dfs


# pylint: disable=too-many-locals
def plot_progress(
    y_data_dfs: Dict[str, pd.DataFrame],
    plot_indiv: bool = True,
    indiv_alpha: float = 0.2,
    percentiles: Optional[Tuple[float, float]] = None,
    percentile_alpha: float = 0.1,
    x_data_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    stats_include_nan: bool = False,
) -> go.Figure:
    """Create plotly figure based on data.

    Parameters
    ----------
    y_data_dfs : dict[str, pandas.DataFrame]
        keys provide labels, values are what to plot
    plot_indiv : bool, default True
        when a DataFrame has more than one column, this function will plot
        the mean. Setting ``plot_indiv`` will show the individual columns
        as well
    indiv_alpha : float, default 0.2
        when ``plot_indiv`` is set, this sets the alpha of the individual lines
    percentiles : Tuple[float, float], optional
        what percentiles to show (low, high)
    percentile_alpha : float, default 0.2
        when ``percentiles`` is set, this sets the alpha of the percentile \
        lines
    x_data_dfs : Dict[str, pandas.DataFrame]
        a provided x axis for each of the lines
    stats_include_nan : bool, default False
        when computing the mean or percentiles, whether to include rows in the dataframe
        that include nan

    Returns
    -------
    fig : plotly.graph_objects.Figure

    """
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS

    assert 0 <= indiv_alpha <= 1
    assert 0 <= percentile_alpha <= 1

    if percentiles is not None:
        assert len(percentiles) == 2 and \
            0 <= percentiles[0] <= 1 and \
            0 <= percentiles[1] <= 1, "percentiles must be between 0 and 1"

    fig = go.Figure(layout=go.Layout(
        showlegend=True,
        hovermode='x',
        hoverlabel_bgcolor='rgba(255, 255, 255, 0.5)',
        font_size=24,
    ))

    def _make_transparency(_color: str, _alpha: float) -> str:
        return f'rgba({_color[4:-1]}, {_alpha})'

    def _plot(_x: Any, _y: Any, **_kwargs: Any) -> go.Scatter:
        _nan_mask = np.logical_or(~np.isnan(_y), ~np.isnan(_x))
        return fig.add_trace(go.Scatter(x=_x[_nan_mask], y=_y[_nan_mask], **_kwargs))

    if x_data_dfs:
        assert set(x_data_dfs) == set(y_data_dfs), (
            f'keys for x_data_dfs and y_data_dfs do not match:\n'
            f'x_data_dfs keys: {", ".join(x_data_dfs)}\n'
            f'y_data_dfs keys: {", ".join(y_data_dfs)}')

    for i, (name, df) in enumerate(y_data_dfs.items()):
        if x_data_dfs:
            x_vals = x_data_dfs[name].mean(axis=1)
        else:
            x_vals = df.index

        color = colors[i % len(colors)]

        if stats_include_nan:
            nan_mask = np.ones_like(x_vals)
        else:
            nan_mask = np.all(~np.isnan(df.values), axis=1)

        _plot(
            x_vals[nan_mask], df[nan_mask].mean(axis=1), name=name, showlegend=True,
            line_color=color, line_width=2, mode='lines',
            hoverlabel_namelength=-1,
            legendgroup=name,
        )

        if plot_indiv:
            clr = _make_transparency(color, indiv_alpha)
            for col in df.columns:
                if x_data_dfs:
                    x_vals = x_data_dfs[name][col]
                _plot(
                    x_vals, df[col], name=col, showlegend=False,
                    line_color=clr, mode='lines',
                    hoverlabel_namelength=-1,
                    hoverinfo='none',
                    legendgroup=name,
                )

        if len(df.columns) > 1 and percentiles:
            fill_clr = _make_transparency(color, percentile_alpha)
            line_clr = _make_transparency(color, 0.0)

            _plot(
                x_vals[nan_mask], df[nan_mask].quantile(percentiles[0], axis=1),
                showlegend=False, line_color=line_clr, mode='lines',
                name=f'{name}-{round(100 * percentiles[0])}%',
                hoverlabel_namelength=-1, hoverinfo='none',
                legendgroup=name,
            )
            _plot(
                x_vals[nan_mask], df[nan_mask].quantile(percentiles[1], axis=1),
                showlegend=False, line_color=line_clr, mode='lines',
                name=f'{name}-{round(100 * percentiles[1])}%',
                hoverlabel_namelength=-1, hoverinfo='none',
                fill='tonexty', fillcolor=fill_clr,
                legendgroup=name,
            )

    return fig


def group_experiments(
    base_dirs: Iterable[Path],
    name_cb: Optional[Callable[[Path], str]] = None,
    exclude_error_experiments: bool = True,
    log_fname: str = "progress.csv",
) -> Dict[str, List[Path]]:
    """Create dict with key (exp name), value (list of indiv experiment dirs).

    ``name_cb`` provides a key for where to place each progress.csv into a
    dictionary. By default, ``name_cb`` returns the experiment name the common
    prefix attached to an experiment so that when num_samples is more than 1,
    each sample is grouped into the same experiment.

    As an example, if under base_dirs we have::


        SAC_DM-v0_a_00000_0/progress.csv
        SAC_DM-v0_a_00001_0/progress.csv
        SAC_DM-v0_b_00000_0/progress.csv

    `group_experiments` will by default output

    .. code-block:: python

        {
            'SAC_DM-v0_a': ['SAC_DM-v0_a_00000_0', 'SAC_DM-v0_a_00001_0'],
            'SAC_DM-v0_b': ['SAC_DM-v0_b_00000_0/progress.csv']
        }

    Alternatively, if given a custom ``name_cb`` to group them all together
    such as

    .. code-block:: python

        def name_cb(abs_path_to_progress_file: Path) -> str:
            return Path(abs_path_to_progress_file).parent.name.split('_')[0]

    `group_experiments` would return

    .. code-block:: python

        {
            'SAC': [
                'SAC_DM-v0_a_00000_0',
                'SAC_DM-v0_a_00001_0',
                'SAC_DM-v0_b_00000_0',
            ],
        }

    See Also
    --------
    * :func:`get_progress`: convert this function output to dataframes
    * :func:`plot_progress`: create plotly figure from output of `get_progress`

    Here is an example of chaining these functions:

    .. code-block:: python
      :linenos:

      base_dir = Path(~/ray/my_experiment).expanduser()
      experiments = rlgear.utils.group_experiments([base_dir])
      dfs = {nm: rlgear.utils.get_progress(exp)
             for nm, exp in experiments.items}
      fig = rlgear.utils.plot_progress(dfs)
      fig.show()

    """
    if name_cb is None:

        def name_cb(abs_path_to_progress_file: Path) -> str:
            return '_'.join(
                Path(abs_path_to_progress_file).parent.name.split('_')[:3])

    assert name_cb is not None

    # this makes sure the same base_dir is not being checked twice while
    # preserving order: https://stackoverflow.com/a/53657523
    base_dirs = list({k: None for k in base_dirs})

    progress_files: List[Path] = []
    for d in base_dirs:
        progress_files += get_files(d, log_fname)

    out: Dict[str, List[Path]] = collections.defaultdict(list)
    error_files: List[str] = []

    # insert so that the dictionary is sorted according to modified time
    for progress_file in progress_files:
        if (progress_file.parent / 'error.txt').exists():
            error_files.append(str(progress_file.parent))

            if exclude_error_experiments:
                continue

        name = name_cb(progress_file)
        if progress_file.parent not in out[name]:
            out[name].append(progress_file.parent)

    if error_files:
        print('Errors detected in runs:')
        print('\n'.join(error_files))

    return out


def smooth(values: Sequence[float], weight: float) -> Sequence[float]:
    """Apply exponential filter to sequence.

    Parameters
    ----------
    values : Sequence[float]
        values to be filtered
    weight : float
        weight for the filter

    """
    smoothed = []
    smoothed.append(values[0])

    # don't smooth the ending nans but do smooth any in between
    L = len(values)
    for i in range(L - 1, -1, -1):
        if not np.isnan(values[i]):
            L = i + 1
            break

    for v in values[1:L]:
        if np.isnan(v):
            smoothed.append(smoothed[-1])
        else:
            if np.isnan(smoothed[-1]):
                smoothed.append(v)
            else:
                smoothed.append(smoothed[-1] * weight + v * (1 - weight))

    if L != len(values):
        smoothed += values[L:].tolist()

    return smoothed
