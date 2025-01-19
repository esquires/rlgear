import tempfile
import sys
import argparse
import socket
import os
import re
import time
import importlib
import pickle
import copy
import shutil
import pprint
import subprocess as sp
from pathlib import Path
from typing import (
    Iterable, Dict, Optional, Any, TypedDict, TypeVar, List, Dict, Union, Tuple
)
import numpy as np

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

    def __repr__(self) -> str:
        return (
            "MetaWriter("
            f"repo_roots={list(self.repo_roots)}, "
            f"files={[f.name for f in self.files]}, "
            f"dirs={[d.name for d in self.dirs]}, "
            f"ignore_patters={self.ignore_patterns}, "
            f"str_data={list(self.str_data)}, "
            f"obs_to_pickle={list(self.objs_to_pickle)}, "
            f"symlink_dir={self.symlink_dir}"
            ")"
        )

    def configure(self) -> None:

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
        self.configure()

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

        meta_dir = (Path(logdir) / 'meta').resolve()

        if meta_dir.exists():
            meta_dir = Path(tempfile.mkdtemp(dir=logdir, prefix='meta-'))
            if self.print_log_dir:
                print(f'existing meta dir found, writing to {meta_dir}')

        meta_dir.mkdir(exist_ok=True, parents=True)

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

        if self.symlink_dir:
            link_tgt = self.symlink_dir / 'latest'
            link_tgt.unlink(missing_ok=True)

            try:
                link_tgt.symlink_to(logdir, target_is_directory=True)
            except FileExistsError:
                pass

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
            inp = repo_dir / f_str
            is_latest = inp.name == "latest" and inp.is_symlink()
            if not is_submodule and not is_latest:
                out = meta_repo_dir / 'repo' / Path(f_str)

                if out.is_dir():
                    out.mkdir(exist_ok=True, parents=True)
                else:
                    out.parent.mkdir(exist_ok=True, parents=True)
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

        found_files = []
        for d in search_dirs:
            temp_found_files = list(Path(d).rglob(fname_path.name))
            if temp_found_files:
                found_files += temp_found_files

        if not found_files:  # pylint: disable=no-else-raise
            raise FileNotFoundError(
                f'could not find {fname} in {search_dirs}')

        if len(found_files) > 1:
            print((
                f"Found multiple files named {fname}:\n{found_files}\n"
                f"Using first entry."
            ))

        return found_files[0]


def get_inputs(
    yaml_file: StrOrPath,
    search_dirs: Union[StrOrPath, Iterable[StrOrPath]]
) -> List[Path]:
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


def parse_inputs(yaml_files: Iterable[StrOrPath]) -> Dict[Any, Any]:
    """Return a dictionary from a list of yaml files.

    This is a wrapper around yaml.safe_load that merges multiple yaml file
    values. When multiple yaml files set the same value the latter takes
    precedence. see also :func:`get_inputs`

    Parameters
    ----------
    yaml_files : list[Path]
        list of yaml files to parse.

    """
    out: Dict[Any, Any] = {}
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


def dict_str2num(d: Dict[Any, Any]) -> Dict[Any, Any]:
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
    search_dirs: Union[StrOrPath, Iterable[StrOrPath]],
    exp_name: str,
    **meta_writer_kwargs: Any,
) -> Tuple[Dict[Any, Any], MetaWriter, Path, List[Path]]:
    """Wraps common function calls when processing yaml files.

    Parameters
    ----------
    yaml_file : StrOrPath
        see :func:`get_inputs`
    search_dirs : StrOrPath | Iterable[StrOrPath]
        see :func:`get_inputs`
    exp_name : str
        used to create a string that can be used as a log directory
    meta_writer_kwargs : Any
        additional arguments for the MetaWriter constructor

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

    if 'repo_roots' in meta_writer_kwargs:
        meta_writer_kwargs['repo_roots'] |= params['log']['repos']
    else:
        meta_writer_kwargs['repo_roots'] = params['log']['repos']

    if 'files' in meta_writer_kwargs:
        meta_writer_kwargs['files'] += inputs
    else:
        meta_writer_kwargs['files'] = inputs

    str_data = {'params.yaml': params, 'host.txt': socket.gethostname()}
    if 'str_data' in meta_writer_kwargs:
        meta_writer_kwargs['str_data'] |= str_data
    else:
        meta_writer_kwargs['str_data'] = str_data

    meta_writer = MetaWriter(**meta_writer_kwargs)

    prefixes = [Path(d).expanduser() for d in params['log']['prefixes']]

    try:
        prefix = next((p for p in prefixes if p.is_dir()))
    except StopIteration as e:
        msg = (
            "None of the following paths in params['log']['prefixes'] exist: "
            f"{', '.join(str(p) for p in prefixes)}. "
        )

        if len(prefixes) == 1:
            msg += f'Run "mkdir -p {prefixes[0]}"'
        else:
            msg += "Run one of the following:\n"
            msg += "\n".join(f"mkdir -p {p}" for p in prefixes)
        raise FileNotFoundError(msg) from e

    log_dir = \
        prefix / params['log']['exp_group'] / Path(yaml_file).stem / exp_name
    return params, meta_writer, log_dir, inputs


def write_metadata(
    yaml_file: StrOrPath,
    search_dirs: Optional[Union[StrOrPath, Iterable[StrOrPath]]] = None,
    out_dir: Optional[Path] = None,
    **meta_writer_kwargs: Any,
) -> Path:
    """Wrapper for reading a yaml file and writing meta data.

    This is most commonly used for postprocessing.

    Parameters
    ----------
    yaml_file : StrOrPath
        see :func:`get_inputs`
    search_dirs : Optional[StrOrPath | Iterable[StrOrPath]]
        see :func:`get_inputs`. if None, searches the current working directory for a
        repo root
    out_dir : Optional[Path]
        where to write the meta data
    """
    yaml_file = Path(yaml_file)

    if out_dir is None:
        out_dir = Path(tempfile.mkdtemp(prefix=f'{yaml_file.stem}-'))

    if search_dirs is None:
        # pylint: disable=import-outside-toplevel
        import git
        search_dirs = git.Repo(Path.cwd(), search_parent_directories=True).working_dir

    meta_writer = from_yaml(
        yaml_file, search_dirs, yaml_file.stem, **meta_writer_kwargs
    )[1]

    meta_writer.write(out_dir)

    return Path(out_dir)


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


def get_files(base_dir: Path, fname: str) -> List[Path]:
    # this is faster than
    # glob.glob(str(_base_dir) + '/**/fname', recursive=True)
    # since once we find a fname file we can stop searching
    # that tree

    def _helper(_base_dir: Path, _out: List[Path]) -> None:
        # passing the output into the function avoids having to create a list
        # for every branch of the tree.
        # having a helper function avoids confusing the signature of the outer
        # function with an output as an input.
        if (_base_dir / fname).exists():
            _out.append(_base_dir / fname)
            return

        for _child in _base_dir.iterdir():
            if _child.is_dir():
                _helper(_child, _out)

    out: List[Path] = []
    _helper(base_dir, out)

    return out


class ImportClassDict(TypedDict):
    cls: str
    kwargs: Dict[str, Any]


def import_class(
    class_info: Union[str, ImportClassDict],
    **kwargs: Any
) -> Any:
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
    kwargs : Any
        additional keyword arguments to be provided to the constructor
        (ignored if ``class_info`` is a str)

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
        cls_kwargs = class_info['kwargs']
        keys = copy.deepcopy(list(cls_kwargs.keys()))
        for key in keys:
            if key.startswith('__preprocess_'):
                cls_kwargs[key.replace('__preprocess_', '')] = \
                    import_class(cls_kwargs[key])
                del cls_kwargs[key]

        try:
            merged_kwargs = {**cls_kwargs, **kwargs}
            return _get_class(class_info['cls'])(**merged_kwargs)
        except Exception as e:
            print(f'could not initialize {class_info}')
            raise e


T = TypeVar('T')


def interp(x: T, x_low: Any, x_high: Any, y_low: Any, y_high: Any) -> T:
    """Linear interpolation."""
    if x_low == x_high:
        return y_low
    else:
        pct = (x - x_low) / (x_high - x_low)
        return y_low + pct * (y_high - y_low)


def add_rlgear_args(parser: argparse.ArgumentParser) \
        -> argparse.ArgumentParser:
    """Add arguments ``yaml_file``, ``exp_name``, and ``--debug`` to parser."""
    parser.add_argument('yaml_file')
    parser.add_argument('exp_name')
    parser.add_argument('--debug', action='store_true')
    return parser


class Profiler:
    def __init__(self) -> None:
        self.beg_times: Dict[Any, float] = {}
        self.end_times: Dict[Any, float] = {}
        self.overall_beg_time = time.perf_counter()
        self.overall_end_time: Optional[float] = None

    def add(self, key: Any) -> Any:  # pylint: disable=no-self-use
        class _Timer:
            def __enter__(self_timer) -> None:  # pylint: disable=no-self-argument
                self.beg_times[key] = time.perf_counter()

            def __exit__(self_timer, *args) -> None:  # pylint: disable=no-self-argument
                self.end_times[key] = time.perf_counter()
        return _Timer()

    def report(self, normalize: bool) -> Dict[Any, float]:
        if self.overall_end_time is None:
            self.overall_end_time = time.perf_counter()

        T = self.overall_end_time - self.overall_beg_time
        raw_times = {k: self.end_times[k] - self.beg_times[k] for k in self.beg_times}

        if not normalize:
            return raw_times
        else:
            normalized_times = {k: v / T for k, v in raw_times.items()}
            return normalized_times


def check_eq_sets(set1: Iterable[Any], set2: Iterable[Any]) -> None:
    set1 = set(set1)
    set2 = set(set2)
    msg = ""
    if set1 - set2:
        msg += f"set1 has extra keys {set1 - set2}. "
    if set2 - set1:
        msg += f"{set2} expects keys {set2 - set1}. "
    assert set1 == set2, msg
