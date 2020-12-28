import dataclasses
from typing import Callable, Any, List

import tensorflow as tf
from kerastuner.engine import base_tuner
from kerastuner import Hyperband

import mlflow.tensorflow


@dataclasses.dataclass
class Config:
    project: str
    experiment_name: str
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


class MlflowHyperband(Hyperband):
    def _build_and_fit_model(self, trial, fit_args, fit_kwargs):
        """For AutoKeras to override.
        DO NOT REMOVE this function. AutoKeras overrides the function to tune
        tf.data preprocessing pipelines, preprocess the dataset to obtain
        the input shape before building the model, adapt preprocessing layers,
        and tune other fit_args and fit_kwargs.
        # Arguments:
            trial: A `Trial` instance that contains the information
                needed to run this trial. `Hyperparameters` can be accessed
                via `trial.hyperparameters`.
            fit_args: Positional arguments passed by `search`.
            fit_kwargs: Keyword arguments passed by `search`.
        # Returns:
            The fit history.
        """
        model = self.hypermodel.build(trial.hyperparameters)
        mlflow.tensorflow.autolog()
        with mlflow.start_run(run_name=self.project_name):
            return model.fit(*fit_args, **fit_kwargs)