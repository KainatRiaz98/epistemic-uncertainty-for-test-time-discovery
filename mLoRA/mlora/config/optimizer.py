from abc import abstractmethod
from typing import Any, Dict, Type
from typing_extensions import override

from .config import DictConfig


class OptimizerConfig(DictConfig):
    lr_: float
    optimizer_: str

    __params_map: Dict[str, str] = {
        "lr_": "lr",
        "optimizer_": "optimizer",
    }

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)

        self.lr_ = float(self.lr_)

    @abstractmethod
    def to_fn_parameters(self) -> Dict[str, str]: ...


class SGDOptimizerConfig(OptimizerConfig):
    momentum_: float

    __params_map: Dict[str, str] = {"momentum_": "momentum"}

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)

        self.momentum_ = float(self.momentum_)

    @override
    def to_fn_parameters(self) -> Dict[str, Any]:
        return {"lr": float(self.lr_), "motentum": float(self.momentum_)}


class AdamWOptimizerConfig(OptimizerConfig):
    __params_map: Dict[str, str] = {
        "beta1_": "beta1",
        "beta2_": "beta2",
        "weight_decay_": "weight_decay",
    }

    # Defaults match PyTorch AdamW so existing mLoRA configs are unaffected
    beta1_: float = 0.9
    beta2_: float = 0.999
    weight_decay_: float = 0.01

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        # Only parse fields that are present; keep defaults otherwise
        present = {k: v for k, v in self.__params_map.items() if v in config}
        self.init(present, config)

        self.beta1_ = float(self.beta1_)
        self.beta2_ = float(self.beta2_)
        self.weight_decay_ = float(self.weight_decay_)

    @override
    def to_fn_parameters(self) -> Dict[str, Any]:
        return {
            "lr": float(self.lr_),
            "betas": (self.beta1_, self.beta2_),
            "weight_decay": self.weight_decay_,
        }


OPTIMIZERCONFIG_CLASS: Dict[str, Type[OptimizerConfig]] = {
    "sgd": SGDOptimizerConfig,
    "adamw": AdamWOptimizerConfig,
}
