import gym

import numpy as np

import rlgear


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
    smoothed_vals = rlgear.utils.smooth(vals, 0.5)
    assert len(smoothed_vals) == 2
    assert smoothed_vals[0] == 1
    assert np.isclose(smoothed_vals[1], 1.5, atol=1e-9)


if __name__ == '__main__':
    test_import_class()
