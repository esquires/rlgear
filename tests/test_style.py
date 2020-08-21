import subprocess
from pathlib import Path


def test_style() -> None:
    python_paths = \
        (Path(__file__).resolve().parent.parent).rglob('*.py')
    python_files = [str(p) for p in python_paths]
    subprocess.check_call(['flake8'] + python_files)
    subprocess.check_call(
        ['pydocstyle'] + python_files + ['--ignore=D1,D203,D213,D416'])
    rcfile = str(Path(__file__).resolve().parent / '.pylintrc')
    subprocess.check_call(
        ['pylint'] + python_files +
        [f'--rcfile={rcfile}', '--score=no', '--reports=no'])
    subprocess.check_call(
        ['mypy'] + python_files +
        ['--disallow-untyped-defs', '--ignore-missing-imports'])


if __name__ == '__main__':
    test_style()
