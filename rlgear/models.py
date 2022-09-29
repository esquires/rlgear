import functools
from typing import \
    Dict, Sequence, Union, Any, Iterable, List, Tuple, Optional, Type

import numpy as np
import gym

import torch
from torch import nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.misc import same_padding
from ray.rllib.policy.rnn_sequencing import add_time_dimension


def xavier_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)  # type: ignore


def init_modules(modules: Iterable[nn.Module]) -> None:
    for m in modules:
        if torch.cuda.is_available():
            m.cuda()
        xavier_init(m)


def make_fc_layers(
        sizes: Sequence[int], dropout_pct: float, act_cls: Type[nn.Module]) \
        -> List[nn.Module]:
    layers: List[nn.Module] = []
    for inp_size, out_size in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(inp_size, out_size))
        if dropout_pct > 0:
            layers.append(nn.Dropout(p=dropout_pct))
        layers.append(act_cls())
    return layers


def make_box(sz: int) -> gym.spaces.Box:
    lim = np.inf * np.ones(sz, dtype=np.float32)
    return gym.spaces.Box(-lim, lim)


def state_helper(module: nn.Module, lstm_cell_size: int) -> List[torch.Tensor]:
    # https://github.com/ray-project/ray/blob/master/rllib/examples/models/rnn_model.py
    # make hidden states on same device as model
    new = module.weight.new  # type: ignore
    h = [
        new(1, lstm_cell_size).zero_().squeeze(0),  # type: ignore
        new(1, lstm_cell_size).zero_().squeeze(0),  # type: ignore
    ]
    return h


IntOrSeq = Union[int, Sequence, np.ndarray]

# this is declared because often we want to allow Union input but this
# is generally a bad idea
# https://github.com/python/mypy/issues/1693
# so this is declared so we can imply meaning without causing mypy errors
SameAsInput = Any


def out_shape(inp_shape: IntOrSeq, kernel: IntOrSeq, stride: IntOrSeq = 1,
              padding: IntOrSeq = 0) -> SameAsInput:
    """Apply convolution arithmetic.

    https://arxiv.org/pdf/1603.07285.pdf
    """
    # handle scalars or numpy arrays
    # https://stackoverflow.com/a/29319864
    i = np.asarray(inp_shape)
    p = np.asarray(padding)
    k = np.asarray(kernel)
    s = np.asarray(stride)
    scalar_input = False
    if i.ndim == 0:
        i = i[np.newaxis]
        scalar_input = True

    o = np.floor((i + 2 * p - k) / s) + 1
    o = o.astype(np.int32)

    return np.squeeze(o) if scalar_input else o


def dqn_cnn(obs_shape: Sequence[int]) -> Tuple[List[nn.Module], List[int]]:
    channels = [obs_shape[-1], 32, 64, 64]
    kernels = [8, 4, 4]
    strides = [4, 2, 2]

    cnn_layers: List[nn.Module] = []
    out_shp = np.asarray(obs_shape[:-1])
    for in_c, out_c, k, s in \
            zip(channels[:-1], channels[1:], kernels, strides):
        out_shp = out_shape(np.asarray(out_shp), k, s)

        cnn_layers.append(nn.Conv2d(in_c, out_c, kernel_size=k, stride=s))
        cnn_layers.append(nn.ReLU())

    return cnn_layers, out_shp.tolist() + [channels[-1]]


# pylint: disable=abstract-method
class TorchModel(TorchModelV2, nn.Module):
    def __init__(self, *args, **kwargs):  # type: ignore
        TorchModelV2.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)
        self._cur_value = None

    def _make_linear_head(self, inp_size: int) -> None:
        self.pi_layer = \
            nn.Linear(inp_size, int(self.num_outputs))  # type: ignore
        self.v_layer = nn.Linear(inp_size, 1)
        init_modules([self.pi_layer, self.v_layer])

    def _forward_helper(
            self, x_pi: torch.Tensor, x_v: Optional[torch.Tensor] = None) \
            -> torch.Tensor:
        logits = self.pi_layer(x_pi)
        self._cur_value = \
            self.v_layer(x_v if x_v is not None else x_pi).squeeze(1)
        self._last_output = logits
        return logits

    @override(TorchModelV2)
    def value_function(self) -> torch.Tensor:
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value


# pylint: disable=too-many-instance-attributes
class FCNet(TorchModel):

    def __init__(
            self, obs_space: gym.Space, action_space: gym.Space,
            num_outputs: int, model_config: dict, name: str, lstm: bool,
            dropout_pct: float, act_cls: Type[nn.Module]):
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name)

        num_inp: int = np.product(obs_space.shape)  # type: ignore
        hiddens = model_config['fcnet_hiddens']

        cls = get_network_class(lstm)
        self.pi_network = \
            cls(num_inp, num_outputs, hiddens, dropout_pct, act_cls)
        self.v_network = cls(num_inp, 1, hiddens, dropout_pct, act_cls)
        init_modules([self.pi_network, self.v_network])

    # pylint: disable=unused-argument
    @override(TorchModelV2)
    def forward(
            self,
            input_dict: Dict[str, torch.Tensor],
            state: List[torch.Tensor],
            seq_lens: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        logits, pi_state_out = self.pi_network(
            input_dict['obs'].float(), state[:2], seq_lens, self.time_major)
        values, v_state_out = self.v_network(
            input_dict['obs'].float(), state[2:], seq_lens, self.time_major)

        self._last_output = logits
        self._cur_value = values.squeeze(1)

        return logits, pi_state_out + v_state_out

    @override(TorchModelV2)
    def value_function(self) -> torch.Tensor:
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    @override(ModelV2)
    def get_initial_state(self) -> List[torch.Tensor]:
        return self.pi_network.get_initial_state() + \
            self.v_network.get_initial_state()


# pylint: disable=abstract-method
class TorchDQNModel(TorchModel):
    def __init__(
            self, obs_space: gym.Space, action_space: gym.Space,
            num_outputs: int, model_config: dict, name: str):
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name)
        cnn_layers, out_shp = dqn_cnn(obs_space.shape)  # type: ignore
        self.cnn = nn.Sequential(*cnn_layers)
        self.fc = nn.Sequential(
            nn.ReLU(), nn.Linear(int(np.prod(out_shp)), 512), nn.ReLU())
        self._make_linear_head(512)
        init_modules([self.cnn, self.fc])

    # pylint: disable=unused-argument
    @override(TorchModelV2)
    def forward(
            self,
            input_dict: Dict[str, torch.Tensor],
            state: List[torch.Tensor],
            seq_lens: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = input_dict['obs'].float().permute(0, 3, 1, 2) / 255.0
        self.emb_cnn = self.cnn(x)
        self.emb_cnn_flat = self.emb_cnn.reshape(self.emb_cnn.size(0), -1)
        self.emb_fc = self.fc(self.emb_cnn_flat)
        return self._forward_helper(self.emb_fc), state


class TorchDQNLSTMModel(TorchModel):
    def __init__(
            self, obs_space: gym.Space, action_space: gym.Space,
            num_outputs: int, model_config: dict, name: str):

        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        self.lstm_cell_size = model_config['lstm_cell_size']
        cnn_layers, out_shp = dqn_cnn(obs_space.shape)  # type: ignore
        self.cnn = nn.Sequential(*cnn_layers)
        self.lstm = nn.LSTM(
            int(np.prod(out_shp)), self.lstm_cell_size, batch_first=True)

        self._make_linear_head(self.lstm_cell_size)

        init_modules([self.cnn, self.lstm])
        self._cur_value = None

    @override(TorchModelV2)
    def forward(
            self,
            input_dict: Dict[str, torch.Tensor],
            state: List[torch.Tensor],
            seq_lens: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        # first apply the cnn
        x = input_dict['obs'].float().permute(0, 3, 1, 2) / 255.0
        x = self.cnn(x)

        # add time
        x_flat = x.view(x.shape[0], -1)

        # pylint: disable=too-many-function-args,missing-kwoa
        x = add_time_dimension(x_flat, seq_lens, "torch")

        # apply lstm
        # pylint: disable=no-member
        x, state_out = self.lstm(
            x, (torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)))

        # pylint: disable=no-member
        x = torch.reshape(x, [-1, self.lstm_cell_size])

        return self._forward_helper(x), [
            torch.squeeze(state_out[0], 0),
            torch.squeeze(state_out[1], 0)]

    @override(ModelV2)
    def get_initial_state(self) -> List[torch.Tensor]:
        return state_helper(self.cnn[-2], self.lstm_cell_size)


class TorchImpalaModel(TorchModel):
    """Implementation of Impala model in pytorch.

    see here:
    https://github.com/deepmind/scalable_agent/blob/master/experiment.py
    """

    def __init__(
            self, obs_space: gym.Space, action_space: gym.Space,
            num_outputs: int, model_config: dict, name: str):
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name)

        self.fc_emb_sz = 512
        self.convs, self.res_blocks, self.cnn_emb_sz = \
            self._cnn(obs_space.shape)  # type: ignore
        self.fc = self._fc(int(np.prod(self.cnn_emb_sz)))
        self._make_linear_head(self.fc_emb_sz)
        init_modules([self.convs, self.res_blocks, self.fc,
                      self.pi_layer, self.v_layer])

    # pylint: disable=unused-argument
    @override(TorchModelV2)
    def forward(
            self,
            input_dict: Dict[str, torch.Tensor],
            state: List[torch.Tensor],
            seq_lens: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        x = input_dict['obs'].float().permute(0, 3, 1, 2) / 255.0
        for conv, res_block_group in zip(self.convs, self.res_blocks):
            x = conv(x)
            for res_block in res_block_group:  # type: ignore
                x = x + res_block(x)

        self.cnn_emb = x
        self.cnn_emb_vec = self.cnn_emb.reshape(self.cnn_emb.size(0), -1)
        self.fc_emb = self.fc(self.cnn_emb_vec)
        return self._forward_helper(self.fc_emb), state

    def _fc(self, inp_sz: int) -> nn.Module:
        return nn.Sequential(
            nn.ReLU(), nn.Linear(inp_sz, self.fc_emb_sz), nn.ReLU())

    # pylint: disable=too-many-locals,no-self-use
    def _cnn(self, obs_shape: Tuple[int, int, int]) \
            -> Tuple[nn.ModuleList, nn.ModuleList, Tuple[int, int, int]]:
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
            nn.Conv2d, kernel_size=kernel_conv, stride=stride_conv)
        conv_padding = functools.partial(
            same_padding, filter_size=[kernel_conv, kernel_conv],
            stride_size=[stride_conv, stride_conv])

        maxpool_padding = functools.partial(
            same_padding, filter_size=[pool_kernel, pool_kernel],
            stride_size=[pool_stride, pool_stride])

        for in_c, out_c in zip(channels[:-1], channels[1:]):

            layers: List[nn.Module] = []
            padding, out_shp = conv_padding(out_shp)
            layers.append(nn.ZeroPad2d(padding))
            layers.append(conv2d(in_c, out_c))

            padding, out_shp = maxpool_padding(out_shp)
            layers.append(nn.ZeroPad2d(padding))
            layers.append(
                nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride))

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


class MLPNet(nn.Module):
    def __init__(
            self, num_inp: int, num_out: int, hiddens: List[int],
            dropout_pct: float, act_cls: Type[nn.Module]):
        super().__init__()

        self.trunk = nn.Sequential(
            *make_fc_layers([num_inp] + hiddens, dropout_pct, act_cls))

        if hiddens:
            self.head = nn.Linear(hiddens[-1], num_out)
        else:
            self.head = nn.Linear(num_inp, num_out)

        self.logits: Optional[torch.Tensor] = None

        init_modules([self.trunk, self.head])

    # pylint: disable=unused-argument
    def forward(
            self,
            x: torch.Tensor,
            state: List[torch.Tensor],
            seq_lens: torch.Tensor,
            time_major: bool) -> Tuple[torch.Tensor, list]:

        if bool(self.trunk):
            self.emb = self.trunk(x)
        else:
            self.emb = x
        self.logits = self.head(self.emb)
        assert self.logits is not None
        return self.logits, []

    # pylint: disable=no-self-use
    def get_initial_state(self) -> List[torch.Tensor]:
        return []


class LSTMNet(nn.Module):
    def __init__(
            self, num_inp: int, num_out: int, hiddens: List[int],
            dropout_pct: float, act_cls: Type[nn.Module]):
        super().__init__()

        if len(hiddens) == 1:
            self.mlp = None
            self.lstm = nn.LSTM(num_inp, hiddens[-1], batch_first=True)
            init_modules([self.lstm])
        else:
            self.mlp = nn.Sequential(*make_fc_layers(
                [num_inp] + hiddens[:-1], dropout_pct, act_cls))
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
            time_major: bool) -> Tuple[torch.Tensor, list]:

        batch_size = x.shape[0]
        max_seq_len = x.shape[0] // seq_lens.shape[0]
        state = [s.view(1, s.shape[0], s.shape[1]) for s in state]

        if self.mlp is not None:
            self.mlp_emb = self.mlp(x)
            x = self.mlp_emb
        else:
            self.mlp_emb = None

        # run through lstm
        x_time = add_time_dimension(
            x, max_seq_len=max_seq_len, framework="torch",
            time_major=time_major)
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
                ih.new(1, self.lstm_size).zero_().squeeze(0)  # type: ignore
            ]


HelperNet = Union[Type[MLPNet], Type[LSTMNet]]


def distribute_state(networks: List[HelperNet], state: List[torch.Tensor]) \
        -> List[List[torch.Tensor]]:
    out: List[List[torch.Tensor]] = []
    idx = 0
    for network in networks:
        if isinstance(network, LSTMNet):
            out.append(state[idx:idx + 2])
            idx += 2
        else:
            out.append([])

    return out


def get_network_class(lstm: bool) -> HelperNet:
    return LSTMNet if lstm else MLPNet
