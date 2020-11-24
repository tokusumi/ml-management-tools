import dataclasses
from typing import Callable, Any, List

import tensorflow as tf
from kerastuner.engine import base_tuner


@dataclasses.dataclass
class Config:
    project: str
    preprocess: Callable[[Any], Any]
    model: Callable[[Any], Any]
    oracle: base_tuner
    logdir: str = "logs/search"
    epochs: int = 3
    objective: str = "val_acc"
    max_epochs: int = 10
    callbacks: List[tf.keras.callbacks.Callback] = dataclasses.field(
        default_factory=lambda: []
    )
