import pickle
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

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


class Saver:
    def __init__(self, interval: int, log_dir: Path, max_num: int):
        self.interval = interval
        self.log_dir = log_dir
        self.max_num = max_num

        self.last_elapsed = 0
        self.save_files: list[tuple[Path, Optional[Path]]] = []

    def save(
        self,
        elapsed: int,
        modules: list[torch.nn.Module | torch.optim.Optimizer],
        pickle_val: Any = None,
    ) -> None:

        if elapsed - self.last_elapsed < self.interval:
            return

        self.last_elapsed = elapsed
        state_dicts = [module.state_dict() for module in modules]

        while len(self.save_files) > self.max_num - 1:
            rm_file, rm_pkl_file = self.save_files.pop(0)
            print(f"removing {rm_file}")
            rm_file.unlink(missing_ok=True)

            if rm_pkl_file is not None:
                print(f"removing {rm_pkl_file}")
                rm_pkl_file.unlink(missing_ok=True)

        save_file = self.log_dir / f"model_{elapsed:06d}"
        print(f"saving to {save_file}")
        torch.save(state_dicts, save_file)

        if pickle_val is not None:
            pickle_file = save_file.with_suffix(".pkl")
            with open(pickle_file, "wb") as f:
                pickle.dump(pickle_val, f)
        else:
            pickle_file = None

        self.save_files.append((save_file, pickle_file))

    @staticmethod
    def load(model_path: Path, modules: list[torch.nn.Module]) -> Any:

        if model_path.is_dir():
            model_paths = [
                p for p in sorted(model_path.glob("model_*")) if not p.suffix == ".pkl"
            ]

            if not model_paths:
                raise RuntimeError(f"could not find models in {model_path}")

            model_path = model_paths[-1]

        state_dicts = torch.load(model_path, weights_only=True)

        for state_dict, module in zip(state_dicts, modules):
            module.load_state_dict(state_dict)

        pickle_path = model_path.with_suffix(".pkl")
        if pickle_path.exists():
            with open(pickle_path, "rb") as f:
                pickle_val = pickle.load(f)
        else:
            pickle_val = None

        return pickle_val
