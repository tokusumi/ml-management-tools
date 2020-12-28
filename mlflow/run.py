import importlib
import sys
import os

import typer
from tensorflow.keras.callbacks import TensorBoard
import mlflow

from configs.base import Config

sys.path.append("configs")


def main(config_name: str):
    typer.echo(f"Experiment: {config_name}")
    config: Config = importlib.import_module(config_name).config

    if not isinstance(config, Config):
        raise ValueError

    mlflow.set_experiment(config.experiment_name)

    tuner = config.oracle(
        config.model,
        objective=config.objective,
        max_epochs=config.max_epochs,
        directory=config.logdir,
        project_name=config.project,
        overwrite=False,
    )

    train_data, test_data = config.preprocess()
    callbacks = config.callbacks + [
        TensorBoard(log_dir=os.path.join(config.logdir, config.project))
    ]
    tuner.search(
        train_data,
        epochs=config.epochs,
        validation_data=test_data,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    typer.run(main)