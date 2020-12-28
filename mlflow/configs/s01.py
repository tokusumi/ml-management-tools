from functools import partial as p

from tensorflow.python.ops.gen_batch_ops import batch

from configs.base import Config, MlflowHyperband
from preprocess import norm
from models import cnn


config = Config(
    project="simple CNN architecture with short training loop",
    experiment_name="small data",
    preprocess=p(norm.make_dataset, dataset="cifar10", batch=2),
    max_epochs=2,
    model=cnn.build_model,
    oracle=MlflowHyperband,
)
