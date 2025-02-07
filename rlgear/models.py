from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch import nn


def xavier_init(m: nn.Module) -> None:
    """Apply xavier initialization to the module."""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)  # type: ignore


def init_modules(modules: Iterable[nn.Module]) -> None:
    """Set module to cuda if available and use xavier initialization."""
    for m in modules:
        if torch.cuda.is_available():
            m.cuda()
        xavier_init(m)


def make_fc_layers(
    sizes: Sequence[int], dropout_pct: float, act_cls: Type[nn.Module]
) -> List[nn.Module]:
    layers: List[nn.Module] = []
    for inp_size, out_size in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(inp_size, out_size))
        if dropout_pct > 0:
            layers.append(nn.Dropout(p=dropout_pct))
        layers.append(act_cls())
    return layers


class MLPNet(nn.Module):
    def __init__(
        self,
        num_inp: int,
        num_out: int,
        hiddens: List[int],
        dropout_pct: float,
        act_cls: Type[nn.Module],
    ):
        super().__init__()

        self.trunk = nn.Sequential(
            *make_fc_layers([num_inp] + hiddens, dropout_pct, act_cls)
        )

        if hiddens:
            self.head = nn.Linear(hiddens[-1], num_out)
        else:
            self.head = nn.Linear(num_inp, num_out)

        self.logits: Optional[torch.Tensor] = None

        init_modules([self.trunk, self.head])

    # pylint: disable=unused-argument
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if bool(self.trunk):
            self.emb = self.trunk(x)
        else:
            self.emb = x
        self.logits = self.head(self.emb)
        assert self.logits is not None
        return self.logits

    # pylint: disable=no-self-use
    def get_initial_state(self) -> List[torch.Tensor]:
        return []


def run_backprop(optimizer: torch.optim.Adam, loss: torch.Tensor) -> None:
    optimizer.zero_grad()
    loss.backward()  # type: ignore
    optimizer.step()
