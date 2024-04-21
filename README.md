# Universal Benchmark for Genetic Disorder Prediction

## Description

This project includes a Python script (`main.py`) that provides a universal benchmarking tool for machine learning models. It includes features for creating train and test datasets, and for training and evaluating decision trees, Keras neural networks, and Torch neural networks based on the Genetic Disorder dataset https://www.kaggle.com/datasets/aibuzz/predict-the-genetic-disorders-datasetof-genomes.

## Features

- Create Train and Test Dataset

- Benchmark Decision Trees

- Benchmark Keras Neural Network

- Benchmark Torch Neural Network

## Installation

To install and run this project, follow these steps:

1. Clone the repository: `git clone <repository_url>`
2. Navigate to the project directory: `cd <project_directory>`
3. Install the required packages: `pip install -r requirements.txt`
4. Run the script: `python main.py`

## Usage

To use the features of `main.py`, you need to pass the feature name as a command-line argument. Here's how to use each feature:

- To create a train and test dataset, run: `python main.py --create_train_test_dataset`
- To benchmark decision trees, run: `python main.py --benchmark_decision_trees`
- To benchmark a Keras neural network, run: `python main.py --benchmark_keras_nn`
- To benchmark a Torch neural network, run: `python main.py --benchmark_torch_nn`

## Output

Each command generates output that is saved in specific directories:

- `python main.py --create_train_test_dataset`: This command generates the train and test datasets which are saved in the `__dataset/` directory as `train_dataset.csv` and `test_dataset.csv`.

- `python main.py --benchmark_decision_trees`: This command benchmarks decision trees and saves the output in the `__output/benchmark_decision_trees/` directory.

- `python main.py --benchmark_keras_nn`: This command benchmarks a Keras neural network. The output, including the model and its performance metrics, is saved in the `__output/benchmark_keras_nn/` directory.

- `python main.py --benchmark_torch_nn`: This command benchmarks a Torch neural network. The output, including the model and its performance metrics, is saved in the `__output/benchmark_torch_nn/` directory.

## License

This project is licensed under the terms of the [MIT License](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt).