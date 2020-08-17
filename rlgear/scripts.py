import argparse

import matplotlib.pyplot as plt

import rlgear


def plot_progress_script() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dirs', nargs='+')
    parser.add_argument('tag')
    parser.add_argument('--percentiles', type=float, nargs=2)
    parser.add_argument('--names', nargs='+')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--only-complete-data', action='store_true')
    parser.add_argument('--show-same-num-timesteps', action='store_true')
    parser.add_argument('--xtick_interval', type=float)

    args = parser.parse_args()

    ax = plt.subplots()[1]
    rlgear.utils.plot_progress(
        ax, args.base_dirs, args.tag, args.names,
        args.only_complete_data, args.show_same_num_timesteps,
        args.percentiles, args.alpha, args.xtick_interval)
    ax.legend()
    plt.show()
