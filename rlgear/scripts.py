import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from .utils import find_tb_fnames, plot_tensorboard


def tensorboard_mean_plot() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dirs', nargs='+')
    parser.add_argument('tag')
    parser.add_argument('--percentiles', type=float, nargs=2)
    parser.add_argument('--max_step', type=int)
    parser.add_argument('--names', nargs='+')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--only_complete_data', action='store_true')
    parser.add_argument('--show_same_num_timesteps', action='store_true')
    parser.add_argument('--xtick_interval', type=float)

    parser.add_argument('--use_files_cache', action='store_true')
    parser.add_argument('--files_cache_fname', default='.rlgear_files.p')
    parser.add_argument('--use_data_cache', action='store_true')
    parser.add_argument('--data_cache_fname', default='.rlgear_data.p')

    args = parser.parse_args()

    ax = plt.subplots()[1]

    if not args.names:
        args.names = [Path(d).parent.name for d in args.base_dirs]

    if args.use_files_cache:
        with open(args.files_cache_fname, 'rb') as f:
            tb_grouped_fnames = pickle.load(f)
    else:
        tb_grouped_fnames = [find_tb_fnames(d) for d in args.base_dirs]
        with open(args.files_cache_fname, 'wb') as f:
            pickle.dump(tb_grouped_fnames, f)

    plot_tensorboard(
        ax, args.tag, tb_grouped_fnames, args.names,
        percentiles=args.percentiles, alpha=args.alpha,
        use_data_cache=args.use_data_cache,
        data_cache_fname=args.data_cache_fname,
        show_same_num_timesteps=args.show_same_num_timesteps,
        max_step=args.max_step)

    if args.xtick_interval:
        ax.xaxis.set_major_locator(mtick.MultipleLocator(args.xtick_interval))

    ax.legend()
    plt.show()
