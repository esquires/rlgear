import copy
import csv
import functools
import numbers
import os
import random
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import same_padding
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override
from rlgear.models import init_modules
from torch import nn

try:
    from torch import Tensor
except ImportError:
    Tensor = None  # type: ignore

import ray
import ray.rllib.algorithms.callbacks
import ray.tune.registry
import ray.tune.trainable.trainable
import ray.tune.utils
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.tune.experiment.trial import Trial
from ray.tune.registry import ENV_CREATOR, _global_registry

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = object  # type: ignore

from .utils import MetaWriter, StrOrPath, import_class


class MetaLoggerCallback(ray.tune.logger.LoggerCallback):
    """Wrap :class:`rlgear.utils.MetaWriter`.

    For use with :class:`ray.tune.logger.LoggerCallback`

    Example
    -------
    .. code-block:: python

        ray.tune.run(
            callbacks=[MetaLoggerCallback(MetaWriter())],
            ...
        )

    """

    def __init__(self, meta_writer: MetaWriter):
        self.meta_writer = meta_writer

    def log_trial_start(self, trial: ray.tune.experiment.trial.Trial) -> None:
        self.meta_writer.write(trial.logdir)


class Filter:
    """Filter extra :class:`ray.tune.logger.LoggerCallback` outputs.

    Parameters
    ----------
    excludes: List[str]
        list of regexes to be compiled via :func:`re.compile`

    """

    def __init__(self, excludes: List[str]):
        self.regexes = [re.compile(e) for e in excludes]

    def __call__(self, d: dict[str, Any]) -> dict[str, Any]:
        flat_result = ray.tune.utils.flatten_dict(d, delimiter="/")

        out = {}

        for key, val in flat_result.items():
            if not any(regex.match(key) for regex in self.regexes):
                out[key] = val

        return out


class AdjPrefix:
    def __init__(self, prefixes: dict[str, str]):
        self.prefixes = prefixes

    def adj(self, val: str) -> str:
        for old_prefix, new_prefix in self.prefixes.items():
            if val.startswith(old_prefix):
                return val.replace(old_prefix, new_prefix, 1)

        return val


class SummaryWriterAdjPrefix(SummaryWriter):
    def __init__(self, prefixes: dict[str, str], *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.adj_prefix = AdjPrefix(prefixes)

    def add_scalar(self, tag: str, *args: Any, **kwargs: Any) -> None:
        new_tag = self.adj_prefix.adj(tag)
        return super().add_scalar(new_tag, *args, **kwargs)


class TBXFilteredLoggerCallback(ray.tune.logger.tensorboardx.TBXLoggerCallback):
    """Wrap :class:`ray.tune.logger.tensorboardx.TBXLoggerCallback`.

    Reduces the output based on the provided :func:`Filter`.
    """

    def __init__(self, filt: Filter, prefixes: dict[str, str]):
        super().__init__()
        self.filt = filt
        self.prefixes = prefixes
        self._summary_writer_cls = self._summary_writer_cls_rm_prefix

    def log_trial_result(
        self,
        iteration: int,
        trial: ray.tune.experiment.trial.Trial,
        result: Dict[str, Any],
    ) -> None:
        super().log_trial_result(iteration, trial, self.filt(result))

    def _summary_writer_cls_rm_prefix(
        self, *args: Any, **kwargs: Any
    ) -> SummaryWriterAdjPrefix:
        return SummaryWriterAdjPrefix(self.prefixes, *args, **kwargs)


class JsonFiltredLoggerCallback(ray.tune.logger.json.JsonLoggerCallback):
    """Wrap :class:`ray.tune.logger.json.JsonLoggerCallback`.

    Reduces the output based on the provided :func:`Filter`.
    """

    def __init__(self, filt: Filter):
        super().__init__()
        self.filt = filt

    def log_trial_result(
        self,
        iteration: int,
        trial: ray.tune.experiment.trial.Trial,
        result: Dict[str, Any],
    ) -> None:
        super().log_trial_result(iteration, trial, self.filt(result))


class CSVFilteredLoggerCallback(ray.tune.logger.csv.CSVLoggerCallback):
    """Wrapper around :class:`ray.tune.logger.csv.CSVLoggerCallback` \
        that reduces the output based on the provided excludes regexes.

    This callback also waits a set number of training iterations before
    freezing the keys as sometimes not all logging items are available
    on the first iteration.
    """

    def __init__(self, wait_iterations: int, excludes: List[str]):
        super().__init__()
        self.wait_iterations = wait_iterations
        self.prior_results: dict[Trial, list[dict[str, Any]]] = defaultdict(list)
        self.keys: dict[Trial, set[str]] = defaultdict(set)
        self.filt = Filter(excludes)
        self.excluded_keys: set[str] = set()

    def log_trial_result(
        self, iteration: int, trial: Trial, result: dict[str, Any]
    ) -> None:

        # see piece of ray.tune.logger.csv.CSVLoggerCallback:log_trial_result
        if trial not in self._trial_files:
            self._setup_trial(trial)

        training_iteration = result["training_iteration"]
        result = self.filt(result)
        self.excluded_keys.update(
            {k for k, r in result.items() if not isinstance(r, numbers.Real)}
        )
        self.keys[trial] |= set(result)

        if not self._trial_csv[trial] and training_iteration >= self.wait_iterations:

            keys = self.keys[trial] - self.excluded_keys

            # see piece of ray.tune.logger.csv.CSVLoggerCallback:log_trial_result
            self._trial_csv[trial] = csv.DictWriter(self._trial_files[trial], keys)
            if not self._trial_continue[trial]:
                self._trial_csv[trial].writeheader()

            # now that we have a csv, write the cached results
            for r in self.prior_results[trial]:
                self.writerow(self._trial_csv[trial], r)

        if self._trial_csv[trial]:
            self.writerow(self._trial_csv[trial], result)
            self._trial_files[trial].flush()
        else:
            # We don't have all the keys yet so don't want to write a header.
            # For now cache the results.
            self.prior_results[trial].append(result)

    @staticmethod
    def writerow(csv_writer: csv.DictWriter, result: dict[Any, Any]) -> None:
        # copied from piece of ray.tune.logger.csv.CSVLoggerCallback:log_trial_result
        csv_writer.writerow({k: result.get(k, np.nan) for k in csv_writer.fieldnames})


def get_trainer(tune_kwargs: dict[Any, Any]) -> ray.tune.trainable.trainable.Trainable:
    trainer_cls = ray.tune.registry.get_trainable_cls(tune_kwargs["run_or_experiment"])
    trainer = trainer_cls(config=tune_kwargs["config"])
    if tune_kwargs["restore"]:
        trainer.restore(str(Path(tune_kwargs["restore"]).expanduser()))
    return trainer


def make_env(env: str, cfg: dict[str, Any], run_check: bool) -> Any:
    env_creator = _global_registry.get(ENV_CREATOR, env)  # type: ignore
    env = env_creator(cfg)
    return env


# pylint: disable=too-many-branches
def make_tune_kwargs(
    params: dict[Any, Any],
    meta_writer: MetaWriter,
    log_dir: StrOrPath,
    debug: bool,
) -> dict[str, Any]:
    # for inputs, see rlgear.utils.from_yaml

    kwargs: dict[str, Any] = {
        "config": {"log_level": "INFO"},
        "local_dir": str(log_dir),
    }

    for blk in params["rllib"]["tune_kwargs_blocks"].split(","):
        kwargs = ray.tune.utils.merge_dicts(kwargs, params["rllib"][blk])

    cfg = kwargs["config"]

    if debug:
        kwargs["local_dir"] = os.path.join(kwargs["local_dir"], "debug")
        cfg["num_workers"] = 0
        cfg["num_gpus"] = 0
        kwargs["max_failures"] = 0
        kwargs["num_samples"] = 1
        if kwargs["verbose"] == 0:
            kwargs["verbose"] = 1
        if cfg["log_level"] in ["ERROR", "WARN"]:
            cfg["log_level"] = "INFO"

    # handle the rllib logger callbacks. These are more complicated because
    # they need to be classes, not the objects. For now just handling
    # the case of a single callback but if needed in the future add support
    # for callbacks.MultiCallbacks
    if "callbacks" in cfg:
        if isinstance(cfg["callbacks"], str):
            cfg["callbacks"] = import_class(cfg["callbacks"])
        else:
            cfg["callbacks"] = ray.rllib.algorithms.callbacks.make_multi_callbacks(
                [import_class(cb) for cb in cfg["callbacks"]]
            )  # type: ignore

    # handle the tune logger callbacks
    if "callbacks" not in kwargs:
        kwargs["callbacks"] = []
    else:
        for i, cb in enumerate(kwargs["callbacks"]):
            if isinstance(cb, dict):
                kwargs["callbacks"][i] = import_class(kwargs["callbacks"][i])

    excludes = params["log"].get("callbacks", {}).get("excludes", [])

    if "csv" in params["log"]["callbacks"]:
        kwargs["callbacks"].append(
            CSVFilteredLoggerCallback(
                params["log"]["callbacks"]["csv"]["wait_iterations"], excludes
            )
        )

    if "tensorboard" in params["log"]["callbacks"]:
        kwargs["callbacks"].append(
            TBXFilteredLoggerCallback(
                Filter(excludes),
                params["log"]["callbacks"]["tensorboard"].get("prefixes", {}),
            )
        )

    if "json" in params["log"]["callbacks"]:
        kwargs["callbacks"].append(JsonFiltredLoggerCallback(Filter(excludes)))

    if "tune_kwargs" not in meta_writer.str_data:
        kwargs_copy = copy.deepcopy(kwargs)
        try:
            del kwargs_copy["config"]["callbacks"]
        except KeyError:
            pass

        meta_writer.str_data["tune_kwargs.yaml"] = kwargs
        meta_writer.objs_to_pickle["tune_kwargs.p"] = kwargs_copy

    kwargs["callbacks"].append(MetaLoggerCallback(meta_writer))

    return kwargs


# pylint: disable=unused-argument
def dirname_creator(trial: ray.tune.experiment.Trial) -> str:
    return trial.trial_id.split("_")[0]


class InfoToCustomMetricsCallback(ray.rllib.algorithms.callbacks.DefaultCallbacks):
    # pylint: disable=arguments-differ,no-self-use
    def on_episode_end(
        self,
        *_: Any,
        episode: Union[Episode, EpisodeV2, Exception],
        **__: Any,
    ) -> None:

        if isinstance(episode, Episode):
            for info in episode._agent_to_last_info.values():
                episode.custom_metrics.update(ray.tune.utils.flatten_dict(info.copy()))
        else:
            episode.custom_metrics.update(
                ray.tune.utils.flatten_dict(episode._last_infos)
            )


def gen_passwd(size: int) -> str:
    """Generate password for ray.init call.

    See
    https://docs.ray.io/en/latest/configure.html?highlight=security#redis-port-authentication

    This function was adapted from https://stackoverflow.com/a/2257449

    Example
    -------

    .. code-block:: python

      ray.init(redis_password=gen_passwd(512))

    Parameters
    ----------
    size : int
        how long the password should be

    """
    # https://stackoverflow.com/a/2257449
    chars = string.ascii_letters + string.digits
    return "".join(random.SystemRandom().choice(chars) for _ in range(size))


def check(x: Tensor, *args: Any, lim: float = 1.0e7, **kwargs: Any) -> None:
    """Raise :py:exc:`ValueError` if ``|x|`` has large or ``nan`` entries.

    Parameters
    ----------
    x : torch.Tensor
        a tensor to be checked for large or ``nan`` values
    args : Any
        these will be printed out in order when ``|x|`` is large or has \
        ``nan`` values
    lim : float, default 1.0e7
        value that defines whether ``|x|`` is large
    kwargs : Any
        these will be printed out in ``|x|`` is large or has ``nan`` values

    """
    import torch  # pylint: disable=import-outside-toplevel

    failed = torch.any(torch.isnan(x))
    isinf = torch.any(torch.isinf(x))
    if np.isinf(lim):
        failed |= isinf
    else:
        failed |= isinf or torch.any(torch.abs(x.float()) >= lim)

    if not failed:
        return

    msg = f"check failed with limit {lim}. x is {x}\n"
    if args:
        msg += "check args are\n"
        for arg in args:
            msg += f"{arg}\n"

    if kwargs:
        msg += "check kwargs are\n"
        for key, val in kwargs.items():
            msg += f"{key}: {val}\n"

    raise ValueError(msg)


# pylint: disable=abstract-method
class TorchModel(TorchModelV2, nn.Module):
    def __init__(self, *args, **kwargs):  # type: ignore
        TorchModelV2.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)
        self._cur_value = None

    def _make_linear_head(self, inp_size: int) -> None:
        self.pi_layer = nn.Linear(inp_size, int(self.num_outputs))  # type: ignore
        self.v_layer = nn.Linear(inp_size, 1)
        init_modules([self.pi_layer, self.v_layer])

    def _forward_helper(
        self, x_pi: torch.Tensor, x_v: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits = self.pi_layer(x_pi)
        self._cur_value = self.v_layer(x_v if x_v is not None else x_pi).squeeze(1)
        self._last_output = logits
        return logits

    @override(TorchModelV2)
    def value_function(self) -> torch.Tensor:
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value


# pylint: disable=too-many-instance-attributes
class FCNet(TorchModel):

    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        num_outputs: int,
        model_config: Dict,
        name: str,
        lstm: bool,
        dropout_pct: float,
        act_cls: Type[nn.Module],
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        num_inp: int = np.product(obs_space.shape)  # type: ignore
        hiddens = model_config["fcnet_hiddens"]

        cls = get_network_class(lstm)
        self.pi_network = cls(num_inp, num_outputs, hiddens, dropout_pct, act_cls)
        self.v_network = cls(num_inp, 1, hiddens, dropout_pct, act_cls)
        init_modules([self.pi_network, self.v_network])

    # pylint: disable=unused-argument
    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        state: List[torch.Tensor],
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        logits, pi_state_out = self.pi_network(
            input_dict["obs"].float(), state[:2], seq_lens, self.time_major
        )
        values, v_state_out = self.v_network(
            input_dict["obs"].float(), state[2:], seq_lens, self.time_major
        )

        self._last_output = logits
        self._cur_value = values.squeeze(1)

        return logits, pi_state_out + v_state_out

    @override(TorchModelV2)
    def value_function(self) -> torch.Tensor:
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    @override(ModelV2)
    def get_initial_state(self) -> List[torch.Tensor]:
        return self.pi_network.get_initial_state() + self.v_network.get_initial_state()


# pylint: disable=abstract-method
class TorchDQNModel(TorchModel):
    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        num_outputs: int,
        model_config: Dict,
        name: str,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        cnn_layers, out_shp = dqn_cnn(obs_space.shape)  # type: ignore
        self.cnn = nn.Sequential(*cnn_layers)
        self.fc = nn.Sequential(
            nn.ReLU(), nn.Linear(int(np.prod(out_shp)), 512), nn.ReLU()
        )
        self._make_linear_head(512)
        init_modules([self.cnn, self.fc])

    # pylint: disable=unused-argument
    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        state: List[torch.Tensor],
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = input_dict["obs"].float().permute(0, 3, 1, 2) / 255.0
        self.emb_cnn = self.cnn(x)
        self.emb_cnn_flat = self.emb_cnn.reshape(self.emb_cnn.size(0), -1)
        self.emb_fc = self.fc(self.emb_cnn_flat)
        return self._forward_helper(self.emb_fc), state


class TorchDQNLSTMModel(TorchModel):
    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        num_outputs: int,
        model_config: Dict,
        name: str,
    ):

        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.lstm_cell_size = model_config["lstm_cell_size"]
        cnn_layers, out_shp = dqn_cnn(obs_space.shape)  # type: ignore
        self.cnn = nn.Sequential(*cnn_layers)
        self.lstm = nn.LSTM(
            int(np.prod(out_shp)), self.lstm_cell_size, batch_first=True
        )

        self._make_linear_head(self.lstm_cell_size)

        init_modules([self.cnn, self.lstm])
        self._cur_value = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        state: List[torch.Tensor],
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        # first apply the cnn
        x = input_dict["obs"].float().permute(0, 3, 1, 2) / 255.0
        x = self.cnn(x)

        # add time
        x_flat = x.view(x.shape[0], -1)

        # pylint: disable=too-many-function-args,missing-kwoa
        x = add_time_dimension(x_flat, seq_lens, "torch")

        # apply lstm
        # pylint: disable=no-member
        x, state_out = self.lstm(
            x, (torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0))
        )

        # pylint: disable=no-member
        x = torch.reshape(x, [-1, self.lstm_cell_size])

        return self._forward_helper(x), [
            torch.squeeze(state_out[0], 0),
            torch.squeeze(state_out[1], 0),
        ]

    @override(ModelV2)
    def get_initial_state(self) -> List[torch.Tensor]:
        return state_helper(self.cnn[-2], self.lstm_cell_size)


class TorchImpalaModel(TorchModel):
    """Implementation of Impala model in pytorch.

    see here:
    https://github.com/deepmind/scalable_agent/blob/master/experiment.py
    """

    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        num_outputs: int,
        model_config: Dict,
        name: str,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.fc_emb_sz = 512
        self.convs, self.res_blocks, self.cnn_emb_sz = self._cnn(
            obs_space.shape
        )  # type: ignore
        self.fc = self._fc(int(np.prod(self.cnn_emb_sz)))
        self._make_linear_head(self.fc_emb_sz)
        init_modules(
            [self.convs, self.res_blocks, self.fc, self.pi_layer, self.v_layer]
        )

    # pylint: disable=unused-argument
    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        state: List[torch.Tensor],
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        x = input_dict["obs"].float().permute(0, 3, 1, 2) / 255.0
        for conv, res_block_group in zip(self.convs, self.res_blocks):
            x = conv(x)
            for res_block in res_block_group:  # type: ignore
                x = x + res_block(x)

        self.cnn_emb = x
        self.cnn_emb_vec = self.cnn_emb.reshape(self.cnn_emb.size(0), -1)
        self.fc_emb = self.fc(self.cnn_emb_vec)
        return self._forward_helper(self.fc_emb), state

    def _fc(self, inp_sz: int) -> nn.Module:
        return nn.Sequential(nn.ReLU(), nn.Linear(inp_sz, self.fc_emb_sz), nn.ReLU())

    # pylint: disable=too-many-locals,no-self-use
    def _cnn(
        self, obs_shape: Tuple[int, int, int]
    ) -> Tuple[nn.ModuleList, nn.ModuleList, Tuple[int, int, int]]:
        channels = [obs_shape[-1], 16, 32, 32, 32]
        kernel_conv = 3
        stride_conv = 1

        pool_kernel = 3
        pool_stride = 2

        num_res_blocks = 2

        out_shp = obs_shape[:-1]
        convs = nn.ModuleList()
        res_blocks = nn.ModuleList()

        conv2d = functools.partial(
            nn.Conv2d, kernel_size=kernel_conv, stride=stride_conv
        )
        conv_padding = functools.partial(
            same_padding,
            filter_size=[kernel_conv, kernel_conv],
            stride_size=[stride_conv, stride_conv],
        )

        maxpool_padding = functools.partial(
            same_padding,
            filter_size=[pool_kernel, pool_kernel],
            stride_size=[pool_stride, pool_stride],
        )

        for in_c, out_c in zip(channels[:-1], channels[1:]):

            layers: List[nn.Module] = []
            padding, out_shp = conv_padding(out_shp)
            layers.append(nn.ZeroPad2d(padding))
            layers.append(conv2d(in_c, out_c))

            padding, out_shp = maxpool_padding(out_shp)
            layers.append(nn.ZeroPad2d(padding))
            layers.append(nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride))

            convs.append(nn.Sequential(*layers))

            res_block_group = nn.ModuleList()
            for _ in range(num_res_blocks):
                padding = conv_padding(out_shp)[0]

                for _ in range(2):
                    res_block: List[nn.Module] = []
                    res_block.append(nn.ReLU())
                    res_block.append(nn.ZeroPad2d(padding))
                    res_block.append(conv2d(out_c, out_c))

                seq_res_block = nn.Sequential(*res_block)
                res_block_group.append(seq_res_block)

            res_blocks.append(res_block_group)

        out_sizes = out_shp + (channels[-1],)
        return convs, res_blocks, out_sizes


class LSTMNet(nn.Module):
    def __init__(
        self,
        num_inp: int,
        num_out: int,
        hiddens: List[int],
        dropout_pct: float,
        act_cls: Type[nn.Module],
    ):
        super().__init__()

        if len(hiddens) == 1:
            self.mlp = None
            self.lstm = nn.LSTM(num_inp, hiddens[-1], batch_first=True)
            init_modules([self.lstm])
        else:
            self.mlp = nn.Sequential(
                *make_fc_layers([num_inp] + hiddens[:-1], dropout_pct, act_cls)
            )
            self.lstm = nn.LSTM(hiddens[-2], hiddens[-1], batch_first=True)
            init_modules([self.mlp, self.lstm])

        self.dropout_pct = dropout_pct
        if self.dropout_pct > 0:
            self.lstm_dropout = nn.Dropout(p=self.dropout_pct)
            init_modules([self.lstm_dropout])

        self.logits: Optional[torch.Tensor] = None
        self.linear = nn.Linear(hiddens[-1], num_out)
        init_modules([self.linear])
        self.lstm_size = hiddens[-1]

    def forward(
        self,
        x: torch.Tensor,
        state: List[torch.Tensor],
        seq_lens: torch.Tensor,
        time_major: bool,
    ) -> Tuple[torch.Tensor, list]:

        batch_size = x.shape[0]
        state = [s.view(1, s.shape[0], s.shape[1]) for s in state]

        if self.mlp is not None:
            self.mlp_emb = self.mlp(x)
            x = self.mlp_emb
        else:
            self.mlp_emb = None

        # run through lstm
        x_time = add_time_dimension(
            x, seq_lens=seq_lens, framework="torch", time_major=time_major
        )
        self.emb, state_out = self.lstm(x_time, state)

        try:
            self.emb = self.emb.view(batch_size, -1)
        except RuntimeError:
            # pylint: disable=no-member
            self.emb = torch.reshape(self.emb, [batch_size, -1])

        if self.dropout_pct > 0:
            self.emb = self.lstm_dropout(self.emb)

        self.logits = self.linear(self.emb)

        state_out = [s.view([s.shape[1], s.shape[2]]) for s in state_out]
        return self.logits, state_out  # type: ignore

    def get_initial_state(self) -> List[torch.Tensor]:
        if self.mlp is not None:
            return state_helper(self.mlp[-2], self.lstm_size)
        else:
            modules = list(self.lstm.modules())[0]
            hh = modules.weight_hh_l0
            ih = modules.weight_ih_l0
            return [
                hh.new(1, self.lstm_size).zero_().squeeze(0),  # type: ignore
                ih.new(1, self.lstm_size).zero_().squeeze(0),  # type: ignore
            ]


def distribute_state(
    networks: List[Any], state: List[torch.Tensor]
) -> List[List[torch.Tensor]]:
    out: List[List[torch.Tensor]] = []
    idx = 0
    for network in networks:
        if isinstance(network, LSTMNet):
            out.append(state[idx : idx + 2])
            idx += 2
        else:
            out.append([])

    return out


def get_network_class(lstm: bool) -> HelperNet:
    return LSTMNet if lstm else MLPNet


def state_helper(module: nn.Module, lstm_cell_size: int) -> List[torch.Tensor]:
    # https://github.com/ray-project/ray/blob/master/rllib/examples/models/rnn_model.py
    # make hidden states on same device as model
    new = module.weight.new  # type: ignore
    h = [
        new(1, lstm_cell_size).zero_().squeeze(0),  # type: ignore
        new(1, lstm_cell_size).zero_().squeeze(0),  # type: ignore
    ]
    return h
