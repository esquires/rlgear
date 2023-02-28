from pathlib import Path

import ray

import rlgear


def test_cartpole() -> None:
    ray.init()
    config_dir = Path(__file__).resolve().parent / 'config'
    params, meta_writer, log_dir = rlgear.utils.from_yaml(
        'test_cartpole.yaml', config_dir, 'test_cartpole')[:3]

    tune_kwargs = rlgear.rllib_utils.make_tune_kwargs(
        params, meta_writer, log_dir, False)

    exp = ray.tune.run(**tune_kwargs)
    trial = exp.get_best_trial('episode_reward_mean', mode='max')
    last_rew = trial.metric_analysis['episode_reward_mean']['last']

    assert last_rew > 50


if __name__ == '__main__':
    test_cartpole()
