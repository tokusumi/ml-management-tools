# No management tool

Configurations:

* Config management: Typer
* ML FW: TensorFlow (keras)
* Hyperparameter search: keras-tuner
* ML management tool: None (TensorBoard)

## How to use

At first, activate virtual environments.

```sh
source env/bin/activate
```

Define experimental configurations in `configs/` with `dataclasses`. (For example, see in `configs/base.py and configs/a01.py`.)

If you call `run.py` with a config python file name as follows (`a01`), the experiment starts:

```sh
python3 run.py a01 
```

Then, logs is saved at `logs/search/simple CNN architecture`. Path is defined at `configs/a01.py` or `configs/base.py`

## How to visualize logs

Use TensorBoard:

```sh
python3 -m tensorboard
```
