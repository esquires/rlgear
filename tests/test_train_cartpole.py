from pathlib import Path

import ray

import rlgear


def test_cartpole() -> None:
    ray.init()
    config_dir = Path(__file__).resolve().parent / 'config'
    tune_kwargs = rlgear.rllib_utils.make_basic_rllib_config(
        'test_cartpole.yaml', 'test_cartpole', config_dir)[-1]

    exp = ray.tune.run(**tune_kwargs)
    trial = exp.get_best_trial('episode_reward_mean')
    last_rew = trial.metric_analysis['episode_reward_mean']['last']

    assert last_rew > 50


if __name__ == '__main__':
    test_cartpole()
