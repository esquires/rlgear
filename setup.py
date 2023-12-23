from setuptools import setup

setup(
    name='rlgear',
    version='0.0.1',
    author='Eric Squires',
    long_description='',
    description='',
    zip_safe=False,
    packages=['rlgear'],
    install_requires=[
        "git-python",
        "tabulate",
        "pandas",
        "plotly",
        "gym",
        "dataclasses",
        # ray packages
        # extra rllib dependencies that don't come through automatically
        "crc32c",
        "requests",
        "dm-tree",
        "lz4",
        "tqdm",
        "scikit-image",
        "tensorboardX"
    ],
    extras_require={
        "test": [
            "pytest",
            "flake8",
            "pylint",
            "pydocstyle",
            "mypy",
            "types-pyyaml"
        ],
    },
)
