import subprocess
from pathlib import Path
from typing import List, Any

import pytest


@pytest.fixture
def python_files() -> List[str]:
    python_paths = \
        (Path(__file__).resolve().parent.parent).rglob('*.py')
    return [str(p) for p in python_paths]


# pylint: disable=redefined-outer-name
def test_flake8(python_files: Any) -> None:
    subprocess.check_call(['flake8'] + python_files)


# pylint: disable=redefined-outer-name
def test_pydocstyle(python_files: Any) -> None:
    subprocess.check_call(
        ['pydocstyle'] + python_files + ['--ignore=D1,D203,D213,D416'])


# pylint: disable=redefined-outer-name
def test_pylint(python_files: Any) -> None:
    rcfile = str(Path(__file__).resolve().parent / '.pylintrc')
    subprocess.check_call(
        ['pylint'] + python_files +
        [f'--rcfile={rcfile}', '--score=no', '--reports=no'])


# pylint: disable=redefined-outer-name
def test_mypy(python_files: Any) -> None:
    subprocess.check_call(
        ['mypy'] + python_files +
        ['--disallow-untyped-defs', '--ignore-missing-imports'])
