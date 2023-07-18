import tempfile
import pickle
from pathlib import Path

import numpy as np
import gym

import rlgear.utils
import rlgear.postprocess


def test_meta_writer() -> None:
    out_dir = Path(tempfile.mkdtemp(prefix='rlgear-'))

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
        dirs=[Path(__file__).parent / 'meta_test'],
        ignore_patterns='*.cpp',
        str_data={'extra_info.txt': 'an extra bit of data'},
        objs_to_pickle={'objects.p': my_var},
    )
    writer.write(out_dir)

    meta_dir = out_dir / 'meta'
    assert meta_dir.exists()

    # test outputs that always exist
    assert (meta_dir / 'args.txt').exists()
    assert (meta_dir / 'requirements.txt').exists()

    # test from repo_roots
    assert (meta_dir / 'rlgear').exists()
    assert (meta_dir / 'rlgear' / 'repo').exists()
    assert (meta_dir / 'rlgear' / 'rlgear_commit.txt').exists()
    assert (meta_dir / 'README.md').exists()

    # test from files
    assert (meta_dir / Path(__file__).name).exists()

    # test from dirs
    assert (meta_dir / 'dirs' / 'meta_test').exists()
    assert (meta_dir / 'dirs' / 'meta_test' / 'file1.py').exists()
    assert (meta_dir / 'dirs' / 'meta_test' / 'file2.yaml').exists()
    assert not (meta_dir / 'dirs' / 'meta_test' / 'file3.cpp').exists()

    # test from str_data
    str_data_path = meta_dir / 'extra_info.txt'
    assert str_data_path.exists()
    with open(str_data_path, 'r', encoding='UTF-8') as f:
        data = f.read()
        assert data.startswith('an extra bit of data')

    # test from objs_to_pickle
    objs_to_pickle_path = meta_dir / 'objects.p'
    assert objs_to_pickle_path.exists()
    with open(objs_to_pickle_path, 'rb') as f:
        assert pickle.load(f) == my_var


def test_import_class() -> None:
    box_cls = rlgear.utils.import_class('gym.spaces.Box')
    assert box_cls == gym.spaces.Box

    box_obj = rlgear.utils.import_class({
        'cls': 'gym.spaces.Box',
        'kwargs': {
            '__preprocess_low':
                {'cls': 'numpy.array', 'kwargs': {'object': [-1, -1]}},
            '__preprocess_high':
                {'cls': 'numpy.array', 'kwargs': {'object': [1, 1]}}
        }
    })
    assert all(box_obj.low == [-1, -1])
    assert all(box_obj.high == [1, 1])


def test_smooth() -> None:
    vals = [1, 2]
    smoothed_vals = rlgear.postprocess.smooth(vals, 0.5)
    assert len(smoothed_vals) == 2
    assert smoothed_vals[0] == 1
    assert np.isclose(smoothed_vals[1], 1.5, atol=1e-9)


if __name__ == '__main__':
    test_import_class()
