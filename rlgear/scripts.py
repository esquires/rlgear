import argparse
import shutil
from pathlib import Path


def copy_trial_data() -> None:
    parser = argparse.ArgumentParser(description=(
        "copy a subtree of a trial for files matching an index. "))
    parser.add_argument('src')
    parser.add_argument('dst')
    parser.add_argument('regex')
    parser.add_argument('-v', action='store_true')
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    src_paths = list(src.rglob(args.regex))

    for src_path in src_paths:
        dst_path = dst / src_path.relative_to(src)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if args.v:
            print(f'{src_path} -> {dst_path}')
        shutil.copy(src_path, dst_path)
