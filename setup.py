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
        # ray packages
        "ray[debug]",
        "ray[tune]",
        "ray[rllib]",
        # extra rllib dependencies that don't come through automatically
        "crc32c",
        "requests",
        "dm-tree",
        "lz4",
    ],
    entry_points={
        'console_scripts':
            ['tensorboard-mean-plot=rlgear.scripts:tensorboard_mean_plot'],
    }
)
