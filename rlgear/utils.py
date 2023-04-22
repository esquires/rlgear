import sys
import socket
import collections
import os
import re
import difflib
import time
import importlib
import pickle
import copy
import shutil
import pprint
import subprocess as sp
from pathlib import Path
from typing import Iterable, List, Union, Dict, Tuple, Optional, Sequence, \
    Any, TypedDict, TypeVar, Callable

import gym
import numpy as np
import pandas as pd

import plotly
import plotly.graph_objects as go

import yaml


StrOrPath = Union[str, Path]


# pylint: disable=too-many-instance-attributes,too-many-arguments
class MetaWriter():
    """Log information required to recreate a run of software.

    Parameters
    ----------
    repo_roots : dict, optional
        key: path to a repo (absolute or relative to Path.cwd())
        value: dict defining a configuration with the following keys

        - base_commit: maps to a sha, branch or tag in the repo
        - check_clean: throw assertion error if the index has changes
        - copy_repo: whether to copy all files not in ``.gitignore`` to
          ``meta/repo_name/repo``

    files : array_like, optional
        files to copy to meta directory.
    dirs : array_like, optional
        directories to copy to the meta/dirs directory
    ignore_patterns : str, optional
        :func:`shutil.ignore_patterns` regex used to ignore items
        (e.g. large files) found in dirs
    str_data : dict, optional
        key: filename to show up under meta directory
        value: text to show up within the file. If the value is not a string,
        the output will be converted to yaml (if a dict) or pprint.pformat.
    objs_to_pickle : dict, optional
        key: filename to show up under meta directory
        value: pickleable object picked into the file
    print_log_dir : bool, default: True
        whether to print the log directory. When used with tune.run,
        the user does not know the log directory when the experiment
        is initiated so this can signal where the logging is occurring.
    symlink_dir: str, default: '.'
        where to create a "latest" symlink to the log directory

    Example
    -------

    Suppose you have a git repo called ``your_repo_name``
    that looks like this::

        file0.py
        my_dir
        ├── file1.py
        ├── file2.yaml
        └── file3.cpp

    where ``foo.py`` has the following contents:

    .. code-block:: python
      :linenos:

      import tempfile
      import rlgear.utils
      out_dir = tempfile.mkdtemp(prefix='rlgear-')
      my_var = 3
      writer = rlgear.utils.MetaWriter(
          repo_roots={
              '.': {
                  'base_commit': 'origin/master',
                  'check_clean': False,
                  'copy_repo': True
              }
          },
          files=[__file__],
          dirs=['my_dir'],
          ignore_patterns='*.cpp'
          str_data={'extra_info.txt': 'an extra bit of data'},
          objs_to_pickle={'objects.p': my_var},
      )
      writer.write(out_dir)

    Running ``python foo.py`` will result in
    the creation a temporary directory ``/tmp/rlgear-randomChars`` with
    a ``meta`` subdirectory. In parenthesis is the relevant constructor
    argument that causes the associated output. In addition there will be
    a softlink called ``latest`` that points to
    ``/tmp/rlgear/rlgear-randomChars``.

    - These files are always produced \
      (i.e., with ``writer = rlgear.utils.MetaWriter()`` below)
        - ``args.txt``: contains command line argument \
          (in this case ``python foo.py``)
        - ``requirements.txt``: output of ``pip freeze``
    - ``repo_root``
        - ``README.md``: a description of how to recreate the repo with saved \
            patches, git diff, and commit information
        - ``your_repo_name``: directory containing a patch, diff, commit, \
            and a full copy of the repo you ran the code from (from the \
            ``repo_roots`` argument)
    - ``files``: ``foo.py`` which is a copy of the ``foo.py`` in your repo
    - ``dirs`` and ``ignore_patterns``: \
        directory called ``my_dir`` containing files ``file1.py`` and \
        ``file2.yaml`` (not ``file3.cpp`` given the ``ignore_patterns``)
    - ``str_data``: ``extra_info.txt`` containing the string "an extra bit \
        of data"
    - ``objs_to_pickle``: ``objects.p`` that contains a python float with \
        value 3

    """

    # pylint: disable=too-many-locals
    def __init__(
        self,
        repo_roots: Optional[Dict[str, Dict[str, Any]]] = None,
        files: Optional[Iterable[StrOrPath]] = None,
        dirs: Optional[Iterable[StrOrPath]] = None,
        ignore_patterns: Optional[Iterable[str]] = None,
        str_data: Optional[Dict[str, Any]] = None,
        objs_to_pickle: Optional[Dict[str, Any]] = None,
        print_log_dir: bool = True,
        symlink_dir: Optional[str] = "."
    ):

        self.repo_roots = {} if repo_roots is None else repo_roots
        self.files = [Path(f).absolute() for f in files] if files else []
        self.dirs = [Path(d).absolute() for d in dirs] if dirs else []
        self.ignore_patterns = [ignore_patterns] \
            if isinstance(ignore_patterns, str) else ignore_patterns
        self.str_data = str_data or {}
        self.objs_to_pickle = \
            objs_to_pickle if objs_to_pickle is not None else {}
        self.print_log_dir = print_log_dir
        self.symlink_dir = \
            Path(symlink_dir).expanduser() if symlink_dir else None

        # https://stackoverflow.com/a/58013217
        self.requirements = sp.check_output(
            [sys.executable, '-m', 'pip', 'freeze']).decode('UTF-8')

        self.orig_dir = Path(os.getcwd())

        self.cmd = self._rel_path(Path(sys.executable)) \
            + " " + " ".join(sys.argv)

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

            for repo_root, repo_config in self.repo_roots.items():

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
                    base = _get(
                        ['git', 'merge-base', 'HEAD', base_commit]).strip()

                    patch = _get(['git', 'format-patch', '--stdout', base])
                    base_commit_hash = _get(
                        ['git', 'rev-parse', '--short', base]).strip()
                    short_commit = _get(
                        ['git', 'rev-parse', '--short', 'HEAD']).strip()

                # pylint: disable=no-member
                except (git_exc.InvalidGitRepositoryError,
                        git_exc.NoSuchPathError,
                        sp.CalledProcessError) as e:
                    print(e)
                    print((f'ignoring the previous git error for {repo_root}. '
                           'This git repo will not be saved.'))
                else:
                    if repo_config['check_clean']:
                        assert not repo.head.commit.diff(None), \
                            ("check_clean is set to True for "
                             f"{repo.common_dir} but the status is not clean")

                    self.git_info[Path(cwd).stem] = {
                        'commit': repo.commit().name_rev,
                        'commit_short': short_commit,
                        'patch': patch,
                        'patch_fname':
                            f'patch_rel_to_{base_commit_hash}.patch',
                        'base_commit_hash': base_commit_hash,
                        'diff': diff,
                        'repo_dir': Path(cwd),
                        'copy_repo': repo_config['copy_repo'],
                        'ignore': repo_config.get('ignore', [])
                    }

    # pylint: disable=too-many-locals,too-many-statements, too-many-branches
    def write(self, logdir: StrOrPath) -> None:
        """Create meta subdirectory and save data.

        This method is separated from the constructor for compatibility
        with the
        `ray.tune.logger.LoggerCallback
        <https://github.com/ray-project/ray/blob/master/python/ray/tune/logger/logger.py>`_
        interface.

        Parameters
        ----------
        logdir : str
            where to log the data

        """

        def _rm_non_pickleable(_obj: Any) -> Any:
            # don't perturb the original object
            if isinstance(_obj, dict) and 'callbacks' in _obj:
                # callbacks are often no pickleable so try to remove these
                _obj_shallow_copy = {
                    k: v for k, v in _obj.items() if k != 'callbacks'}
                return _obj_shallow_copy
            else:
                return _obj

        def _to_str(_obj: Any) -> str:
            if isinstance(_obj, str):
                return _obj
            elif isinstance(data, dict):
                try:
                    return yaml.dump(_obj)
                except TypeError:
                    _obj = _rm_non_pickleable(_obj)
                    try:
                        return yaml.dump(_obj)
                    except TypeError:
                        return pprint.pformat(_obj)
            else:
                return pprint.pformat(_obj)

        if self.print_log_dir:
            print(f'log dir: {logdir}')

        if self.symlink_dir:
            link_tgt = self.symlink_dir / 'latest'
            link_tgt.unlink(missing_ok=True)

            try:
                link_tgt.symlink_to(logdir, target_is_directory=True)
            except FileExistsError:
                pass

        meta_dir = (Path(logdir) / 'meta').resolve()
        meta_dir.mkdir(exist_ok=True)

        for fname_str, data in self.str_data.items():
            with open(meta_dir / fname_str, 'w', encoding='UTF-8') as f:
                f.write(_to_str(data))

        for fname_str, data in self.objs_to_pickle.items():
            with open(meta_dir / fname_str, 'wb') as fp:
                try:
                    pickle.dump(data, fp)
                except (RuntimeError, pickle.PickleError):
                    data = _rm_non_pickleable(data)
                    pickle.dump(data, fp)

        for fname in self.files:
            shutil.copy2(fname, meta_dir)

        if self.dirs:
            (meta_dir / 'dirs').mkdir(exist_ok=True)
            ignore = shutil.ignore_patterns(*self.ignore_patterns) \
                if self.ignore_patterns else None
            for d in self.dirs:
                shutil.copytree(
                    d, meta_dir / 'dirs' / d.name, ignore=ignore)

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

            if repo_data['diff']:
                diff_file = meta_repo_dir / (
                    repo_name + f'_diff_on_{repo_data["commit_short"]}.diff')
                with open(diff_file, 'w', encoding='UTF-8') as f:
                    f.write(repo_data['diff'])

            if repo_data['patch']:
                patch_file = meta_repo_dir / repo_data['patch_fname']
                with open(patch_file, 'w', encoding='UTF-8') as f:
                    f.write(repo_data['patch'])

            if repo_data['copy_repo']:
                self._copy_repo(repo_data['repo_dir'], meta_repo_dir)

        if self.git_info:
            self._create_readme(
                self.git_info, meta_dir, self.cmd, self.orig_dir)

    @staticmethod
    def _copy_repo(repo_dir: Path, meta_repo_dir: Path) -> None:
        ct = 0
        max_tries = 10

        while ct < max_tries:
            # the subprocess calls can fail when other jobs are simultaneously
            # calling so try a few times after a delay. If it still can't do
            # the subprocess call then exit
            try:
                files = set(sp.check_output(
                    ['git', 'ls-files'], cwd=str(repo_dir)
                ).decode('UTF-8').splitlines())

                status_files = sp.check_output(
                    ['git', 'status', '-s', '--untracked-files=all'],
                    cwd=str(repo_dir)
                ).decode('UTF-8').splitlines()
                break
            except sp.CalledProcessError:
                time.sleep(1)

        if ct == max_tries:
            print(f'could not determine files for {repo_dir}')
            return

        for f_str in status_files:
            if f_str.startswith('??'):
                # untracked file so add
                files.add(f_str.split(' ')[-1])
            elif f_str.startswith(' D '):
                # deleted file that has not been committed.
                # remove from copy list since it does not exist and
                # can't be copied
                files.remove(f_str.split(' ')[-1])

        for f_str in files:
            is_submodule = (repo_dir / f_str / '.git').exists()
            if not is_submodule:
                out = meta_repo_dir / 'repo' / Path(f_str)
                out.parent.mkdir(exist_ok=True, parents=True)
                inp = (repo_dir / Path(f_str)).resolve()
                shutil.copy2(inp, out)

    @staticmethod
    def _create_readme(
        git_info: Dict[str, Any],
        meta_dir: Path,
        cmd: str,
        orig_dir: Path,
    ) -> None:
        msg = 'You can recreate the experiment as follows:\n\n```bash\n'

        for repo_name, repo_data in git_info.items():
            d = meta_dir / repo_name

            msg += f'# {repo_name}: '
            if not repo_data['patch'] and not repo_data['diff']:
                msg += ('no commits or changes relative to '
                        f'{repo_data["base_commit_hash"]} so just reset\n')
            elif not repo_data['patch'] and repo_data['diff']:
                msg += ('no commits relative to '
                        f'{repo_data["base_commit_hash"]} so reset and '
                        'apply the diff\n')
            elif repo_data['patch'] and not repo_data['diff']:
                msg += ('commits relative to '
                        f'{repo_data["base_commit_hash"]} but no diff so '
                        'reset and apply the patch\n')
            else:
                msg += 'reset, apply patch/diff\n'
            msg += f"cd {MetaWriter._rel_path(repo_data['repo_dir'])}\n"
            msg += f"git reset --hard {repo_data['base_commit_hash']}"
            msg += ("  # probably want to git stash or "
                    "otherwise save before this line\n")

            diff_file_str = MetaWriter._rel_path(d / (
                repo_name + f'_diff_on_{repo_data["commit_short"]}.diff'))
            patch_file = MetaWriter._rel_path(d / repo_data['patch_fname'])

            if repo_data['patch']:
                msg += f"git am -3 {patch_file}\n"

            if repo_data['diff']:
                msg += f"git apply {diff_file_str}\n\n"
            else:
                msg += '\n'

        msg += (
            '# run experiment (see also requirements.txt for dependencies)\n'
            f'cd {MetaWriter._rel_path(orig_dir)}\n'
            f'{cmd}\n')
        msg += '```'

        with open(meta_dir / 'README.md', 'w', encoding='UTF-8') as f:
            f.write(msg)

    @staticmethod
    def _rel_path(path: Path) -> str:

        try:
            return f'~/{Path(path).relative_to(Path.home())}'
        except ValueError:
            return str(path)


def find_filepath(
    fname: StrOrPath,
    search_dirs: Union[StrOrPath, Iterable[StrOrPath]]
) -> Path:
    """Return first path with fname under search_dirs.

    Raises :py:exc:`StopIteration` if ``fname`` is not found.

    Parameters
    ----------
    fname : str or pathlib.Path
        the filename to look for
    search_dirs : iterable of Paths
        the directories to search

    Returns
    -------
    path : pathlib.Path
        the path to the filename

    """
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
    search_dirs: Union[StrOrPath, Iterable[StrOrPath]]
) -> list[Path]:
    """Recursively find inputs in a yaml_file.

    Example
    -------

    Suppose yaml_1, yaml_2, and yaml_3 files have the following content

    .. code-block:: yaml

        __inputs__: [yaml_2, yaml_3]  # yaml_1
        __inputs__: [yaml_4]  # yaml_2
        __inputs__: [yaml_5]  # yaml_3

    Then ``get_inputs("yaml_1", search_dirs)`` will return

    .. code-block:: python

        ['yaml_4', 'yaml_2', 'yaml_5', 'yaml_3', 'yaml_1']

    Parameters
    ----------
    yaml_file : pathlib.Path
        path to the top level yaml file
    search_dirs : StrOrPath
        the directories to search

    Returns
    -------
    yaml_files : list[pathlib.Path]
        list of yaml files that should be recursively read when parsing
        (see :func:`parse_inputs`)

    """
    inputs = []

    def _get_inputs(fname: StrOrPath) -> None:
        filepath = find_filepath(fname, search_dirs)
        with open(filepath, 'r', encoding='UTF-8') as f:
            params = yaml.safe_load(f)

        if params is not None:
            temp_inputs = params.get('__inputs__', [])

            if isinstance(temp_inputs, str):
                temp_inputs = [temp_inputs]

            for inp in temp_inputs:
                _get_inputs(inp)

            inputs.append(filepath)

    _get_inputs(yaml_file)
    return inputs


def parse_inputs(yaml_files: Iterable[StrOrPath]) -> dict[Any, Any]:
    """Return a dictionary from a list of yaml files.

    This is a wrapper around yaml.safe_load that merges multiple yaml file
    values. When multiple yaml files set the same value the latter takes
    precedence. see also :func:`get_inputs`

    Parameters
    ----------
    yaml_files : list[Path]
        list of yaml files to parse.

    """
    out: dict[Any, Any] = {}
    for inp in yaml_files:
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


def dict_str2num(d: dict[Any, Any]) -> dict[Any, Any]:
    """Given a dictionary, recursively convert strings to numbers."""
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


def from_yaml(
    yaml_file: StrOrPath,
    search_dirs: StrOrPath | Iterable[StrOrPath],
    exp_name: str,
) -> tuple[dict[Any, Any], MetaWriter, Path, list[Path]]:
    """Convert a yaml_file into a dict.

    Parameters
    ----------
    yaml_file : StrOrPath
        see :func:`get_inputs`
    search_dirs : StrOrPath | Iterable[StrOrPath]
        see :func:`get_inputs`
    exp_name : str
        used to create a string that can be used as a log directory

    Returns
    -------
    params : dict
        parameters from the yaml_file after recursively looking in files
        listed under __inputs__ (see :func:`get_inputs` and
        :func:`parse_inputs`)
    meta_writer : MetaWriter
        an initial version of a meta writer with information from the yaml_file
    log_dir : Path
        `prefix / exp_group / yaml_fname / exp_name`
        where prefix and exp_group are found under the log section of the
        yaml_file
    inputs : list[Path]
        the input yaml files used to create the parameters

    """
    inputs = get_inputs(yaml_file, search_dirs)
    params = dict_str2num(parse_inputs(inputs))
    meta_writer = MetaWriter(
        repo_roots=params['log']['repos'],
        files=inputs,
        str_data={
            'params.yaml': params,
            'host.txt': socket.gethostname()
        }
    )

    prefix = next((Path(d).expanduser() for d in params['log']['prefixes']
                   if Path(d).expanduser().is_dir()))
    log_dir = \
        prefix / params['log']['exp_group'] / Path(yaml_file).stem / exp_name
    return params, meta_writer, log_dir, inputs


def get_latest_checkpoint(ckpt_root_dir: StrOrPath) -> str:
    """Return the latest checkpoint subdirectory given a root directory.

    Parameters
    ----------
    ckpt_root_dir : StrOrPath
        root directory to search (e.g. the log directory from tune)

    Returns
    -------
    ckpt_dir : str
        path to the directory containing the latest checkpoint

    """
    ckpts = [str(c) for c in Path(ckpt_root_dir).rglob('*checkpoint-*')
             if 'meta' not in str(c)]
    r = re.compile(r'checkpoint_(\d+)')
    ckpt_nums = [int(r.search(c).group(1)) for c in ckpts]  # type: ignore
    return ckpts[np.argmax(ckpt_nums)]


def group_experiments(
    base_dirs: Iterable[Path],
    name_cb: Optional[Callable[[Path], str]] = None,
    exclude_error_experiments: bool = True
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
    def _get_progress_files(_base_dir: Path, _out: list[Path]) -> None:
        # this is faster than
        # glob.glob(str(_base_dir) + '/**/progress.csv', recursive=True)
        # since once we find a progress.csv file we can stop searching
        # that tree

        _dirs = []
        for _child in _base_dir.iterdir():
            if _child.name == 'progress.csv':
                _out.append(_child)
                return
            elif _child.is_dir():
                _dirs.append(_child)

        for _dir in _dirs:
            _get_progress_files(_dir, _out)

    if name_cb is None:

        def name_cb(abs_path_to_progress_file: Path) -> str:
            return '_'.join(
                Path(abs_path_to_progress_file).parent.name.split('_')[:3])

    assert name_cb is not None

    progress_files: List[Path] = []
    for d in base_dirs:
        _get_progress_files(d, progress_files)

    out: Dict[str, List[Path]] = collections.defaultdict(list)
    error_files: List[str] = []

    # insert so that the dictionary is sorted according to modified time
    progress_files = sorted(progress_files, key=lambda p: p.stat().st_mtime)
    for progress_file in progress_files:
        if (progress_file.parent / 'error.txt').exists():
            error_files.append(str(progress_file.parent))

            if exclude_error_experiments:
                continue

        out[name_cb(progress_file)].append(progress_file.parent)

    if error_files:
        print('Errors detected in runs:')
        print('\n'.join(error_files))

    return {k: sorted(v) for k, v in out.items()}


# pylint: disable=too-many-locals
def get_progress(
    experiments: Iterable[Path],
    x_tag: str = 'timesteps_total',
    tag: str = 'episode_reward_mean',
    only_complete_data: bool = False,
    max_x: Optional[Any] = None,
    names: Optional[Sequence[str]] = None
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
    only_complete_data : bool (default False)
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
        _suggestions = difflib.get_close_matches(_word, _possibilities)

        if _suggestions:
            print(f'suggestions for "{_word}": {", ".join(_suggestions)}')

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
        try:
            df = pd.read_csv(exp / 'progress.csv', low_memory=False)
        except pd.errors.EmptyDataError:
            print(f'{exp} has empty progress.csv, skipping')
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

        if names:
            filtered_names.append(names[i])
        else:
            try:
                filtered_names.append(exp.name.split('_')[3])
            except IndexError:
                filtered_names.append(exp.name)

        dfs.append(df)

    if only_complete_data:
        _shorten_dfs(dfs)

    if max_x is not None:
        _shorten_dfs(dfs)

    return _merge_dfs(dfs, filtered_names) if dfs else None, dfs


# pylint: disable=too-many-locals
def plot_progress(
    y_data_dfs: Dict[str, pd.DataFrame],
    plot_indiv: bool = True,
    indiv_alpha: float = 0.2,
    percentiles: Optional[Tuple[float, float]] = None,
    percentile_alpha: float = 0.1,
    x_data_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    sort_x_vals: bool = True,
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
    sort_x_vals : bool, default = True
        whether to sort the x values in the plots

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
        if sort_x_vals:
            _idxs = np.argsort(_x)
            _x = np.asarray(_x)[_idxs]
            _y = np.asarray(_y)[_idxs]

        return fig.add_trace(go.Scatter(x=_x, y=_y, **_kwargs))

    if x_data_dfs:
        assert set(x_data_dfs) == set(y_data_dfs), (
            f'keys for x_data_dfs and y_data_dfs do not match:\n'
            f'x_data_dfs keys: {", ".join(x_data_dfs)}\n'
            f'y_data_dfs keys: {", ".join(y_data_dfs)}')

    for i, (name, df) in enumerate(y_data_dfs.items()):
        if x_data_dfs:
            x_df = df.index
            mask = ~np.isnan(x_df.mean(axis=1)) & ~np.isnan(df.mean(axis=1))
            x_df = x_df[mask]
            df = df[mask]
            x_vals = x_df.mean(axis=1)
        else:
            x_vals = df.index

        color = colors[i % len(colors)]

        _plot(
            x_vals, df.mean(axis=1), name=name, showlegend=True,
            line_color=color, line_width=2, mode='lines',
            hoverlabel_namelength=-1,
            legendgroup=name,
        )

        if plot_indiv:
            clr = _make_transparency(color, indiv_alpha)
            for col in df.columns:
                _plot(
                    x_df[col], df[col], name=col, showlegend=False,
                    line_color=clr, mode='lines',
                    hoverlabel_namelength=-1,
                    hoverinfo='none',
                    legendgroup=name,
                )

        if len(df.columns) > 1 and percentiles:
            fill_clr = _make_transparency(color, percentile_alpha)
            line_clr = _make_transparency(color, 0.0)

            _plot(
                x_vals, df.quantile(percentiles[0], axis=1),
                showlegend=False, line_color=line_clr, mode='lines',
                name=f'{name}-{round(100 * percentiles[0])}%',
                hoverlabel_namelength=-1, hoverinfo='none',
                legendgroup=name,
            )
            _plot(
                x_vals, df.quantile(percentiles[1], axis=1),
                showlegend=False, line_color=line_clr, mode='lines',
                name=f'{name}-{round(100 * percentiles[1])}%',
                hoverlabel_namelength=-1, hoverinfo='none',
                fill='tonexty', fillcolor=fill_clr,
                legendgroup=name,
            )

    return fig


class ImportClassDict(TypedDict):
    cls: str
    kwargs: Dict[str, Any]


def import_class(class_info: Union[str, ImportClassDict]) -> Any:
    """Convert a string or dict to an object.

    Example
    -------

    .. code-block:: python

        # using a list as class_info (outputs a class, not an object)
        zeros_class = import_class('numpy.zeros')  # <function numpy.zeros>
        zeros_obj = zeros_class(5)  # array([0., 0., 0., 0., 0.])
        # using a dict as class_info (outputs an object)
        zeros_obj = import_class(
            {'cls': 'numpy.zeros', 'kwargs': {'shape': (5)}}
        )  # array([0., 0., 0., 0., 0.])

    Parameters
    ----------
    class_info : str or dict = {cls: str, kwargs: dict}
        parameters to import

    Returns
    -------
    obj : Any

    """
    def _get_class(class_str: str) -> Any:
        _split = class_str.split('.')
        try:
            _module = importlib.import_module('.'.join(_split[:-1]))
            return getattr(_module, _split[-1])
        except Exception:  # pylint: disable=broad-exception-caught
            # e.g. when an object contains another object (e.g. a staticmethod
            # within a class)
            try:
                _module = importlib.import_module('.'.join(_split[:-2]))
                return getattr(getattr(_module, _split[-2]), _split[-1])
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f'could not initialize {class_info}')
                raise e

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
    for v in values[1:]:
        if np.isnan(v):
            smoothed.append(smoothed[-1])
        else:
            if np.isnan(smoothed[-1]):
                smoothed.append(v)
            else:
                smoothed.append(smoothed[-1] * weight + v * (1 - weight))

    return smoothed


T = TypeVar('T')


def interp(x: T, x_low: Any, x_high: Any, y_low: Any, y_high: Any) -> T:
    """Linear interpolation."""
    if x_low == x_high:
        return y_low
    else:
        pct = (x - x_low) / (x_high - x_low)
        return y_low + pct * (y_high - y_low)


class GymLegacyAPIWrapper(gym.Wrapper):
    def step(  # type: ignore
        self,
        action: Any
    ) -> Tuple[Any, float, bool, Dict[Any, Any]]:
        out = self.env.step(action)

        if len(out) == 5:
            obs, rew, terminated, truncated, info = out  # type: ignore
            done = terminated or truncated  # type: ignore
        else:
            obs, rew, done, info = out  # type: ignore

        return obs, rew, done, info

    def reset(self, **kwargs: Any) -> Any:
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and \
                len(out) == 2 and \
                isinstance(out[1], dict):
            out = out[0]

        return out
