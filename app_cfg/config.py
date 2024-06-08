

class Config():
    
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def get_config():
        
        config = {
            'GLOBAL_SEED': 42,
            'RAW_DATASET_PATH': '__dataset/genetic_disorder_dataset.csv',
            'TEST_DATASET_PATH': '__dataset/test_dataset.csv',
            'TRAIN_DATASET_PATH': '__dataset/train_dataset.csv',
            'DATASET_DISTRIBUTION_PLOT_PATH': '__output/dataset_info/dataset_distribution.png',
            'DATASET_LOG_PATH': '__output/dataset_info/dataset_distribution.log',
            'BENCHMARK_DECISION_TREES_PATH': '__output/benchmark_decision_trees/benchmark_decision_trees.csv',
            'BENCHMARK_KERAS_NN_PATH': '__output/benchmark_keras_nn/benchmark_keras_nn.csv',
            'BENCHMARK_TORCH_NN_PATH': '__output/benchmark_torch_nn/benchmark_torch_nn.csv',
            'BENCHMARK_KERAS_NN_TRAIN_GRAPH_PATH': '__output/benchmark_keras_nn/graphs',
            'BENCHMARK_TORCH_NN_TRAIN_GRAPH_PATH': '__output/benchmark_torch_nn/graphs'
        }
        
        return config