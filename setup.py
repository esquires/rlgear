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
        "matplotlib",
        "gym",
        "dataclasses",
        # ray packages
        "ray>=1.6.0",
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
            "types-PyYAML",
        ],
    },
    entry_points={
        'console_scripts':
            ['copy-trial-data=rlgear.scripts:copy_trial_data'],
    }
)
