from functools import partial as p

from kerastuner import Hyperband
from tensorflow.python.ops.gen_batch_ops import batch

from configs.base import Config
from preprocess import norm
from models import cnn


config = Config(
    project="simple CNN architecture with short training loop",
    preprocess=p(norm.make_dataset, dataset="cifar10", batch=2),
    max_epochs=2,
    model=cnn.build_model,
    oracle=Hyperband,
)
