# MLflow

Configurations:

* Config management: Typer
* ML FW: TensorFlow (keras)
* Hyperparameter search: keras-tuner
* ML management tool: MLflow

## How to use

At first, activate virtual environments.

```sh
python3 -m venv env
source env/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

Define experimental configurations in `configs/` with `dataclasses`. (For example, see in `configs/base.py and configs/a01.py`.)

If you call `run.py` with a config python file name as follows (`a01`), the experiment starts:

```sh
python3 run.py a01 
```

Then, logs is saved at `logs/search/simple CNN architecture` and `mlruns/`. Path is defined at `configs/a01.py` or `configs/base.py`

## How to visualize logs

Use MLflow UI:

```sh
mlflow ui
```

## Implementation details

Only add the followings lines in `main` function in `run.py` in `pure/run.py`:

```python
import mlflow
mlflow.set_experiment(config.experiment_name)
```
