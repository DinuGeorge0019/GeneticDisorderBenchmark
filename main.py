

# standard library imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import numpy as np
import tensorflow as tf
import random
import torch

# local library specific imports
from app_cfg import Config
from app_src import DatasetBuilder, DecisionTreeEvaluator, KerasNNEvaluator, PyTorchNNEvaluator

# get global configuration
CONFIG = Config.get_config()


def __seed_everything(seed):
    """
    Seed all random number generators and set torch and CUDA environment for reproducibility.
    Args:
        seed (int): The seed value to use for seeding the random number generators.
    Returns:
        None
    """
    random.seed(seed)  # Seed the Python random module
    os.environ['PYTHONHASHSEED'] = str(seed)  # Seed the hash function used in Python
    np.random.seed(seed)  # Seed the NumPy random module
    tf.random.set_seed(seed)
    torch.manual_seed(seed)  # Seed the PyTorch random module for CPU
    torch.cuda.manual_seed(seed)  # Seed the PyTorch random module for CUDA
    torch.cuda.manual_seed_all(seed)  # Seed the random module for all GPUs
    torch.backends.cudnn.deterministic = True  # Set CuDNN to deterministic mode for reproducibility
    torch.backends.cudnn.benchmark = True  # Enable CuDNN benchmarking for faster training

    return None


def main():
    __seed_everything(CONFIG['GLOBAL_SEED']) 
    
    parser = argparse.ArgumentParser()

    datasetBuilder = DatasetBuilder()
    decisionTreeEvaluator = DecisionTreeEvaluator()
    kerasNNEvaluator = KerasNNEvaluator()
    torchNNEvaluator = PyTorchNNEvaluator()
    
    arguments = [
        ("create_train_test_dataset", datasetBuilder.create_dataset, "Build train and test dataset"),
        ("benchmark_decision_trees", decisionTreeEvaluator.benchmark_models, "Train and Evaluate decision trees"),
        ("benchmark_keras_nn", kerasNNEvaluator.benchmark_model, "Train and Evaluate Keras Neural Network"),
        ("benchmark_torch_nn", torchNNEvaluator.benchmark_model, "Train and Evaluate Torch Neural Network")
    ]

    for arg, _, description in arguments:
        if arg == "create_train_test_dataset":
            parser.add_argument(f'--{arg}', action='store_true', help=description)
        else:
            parser.add_argument(f'--{arg}', action='store_true', help=description)

    params = parser.parse_args()
    for arg, fun, _ in arguments:
        if hasattr(params, arg) and getattr(params, arg):
            print(f"Executing {arg}")
            if arg == "create_train_test_dataset":
                fun()
            else:
                fun()


if __name__ == '__main__':
    main()
