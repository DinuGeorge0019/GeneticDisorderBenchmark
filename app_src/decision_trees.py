
import os
import pandas as pd
import numpy as np
import time
import psutil
import GPUtil
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, recall_score, mean_squared_error, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV

from app_cfg import Config
from .model_evaluator import ModelEvaluator

CONFIG = Config.get_config()

class DecisionTreeWrapper():
    def __init__(self, estimator) -> None:
        self.estimator = estimator
        self.training_time = None
        self.training_ram_used = None
        self.training_cpu_load_used = None
        self.training_gpu_load_used = None
        self.training_gpu_ram_used = None
        self.testing_time = None
        self.testing_ram_used = None
        self.testing_cpu_load_used = None
        self.testing_gpu_load_used = None
        self.testing_gpu_ram_used = None
        
    def fit(self, X, y=None, **kwargs):
        # Create a Process object for the current process
        process = psutil.Process(os.getpid())
        gpus = GPUtil.getGPUs()
        gpu = gpus[0]
        
        # Before training
        mem_before = process.memory_info().rss / (1024.0 ** 3)
        cpu_before = process.cpu_percent(interval=1)
        gpu_load_before = gpu.load*100
        gpu_ram_before = gpu.memoryUsed
        
        start_time = time.time()

        # Training
        fit_result = self.estimator.fit(X, y, **kwargs)
        
        # After training
        end_time = time.time()
        mem_after = process.memory_info().rss / (1024.0 ** 3)
        cpu_after = process.cpu_percent(interval=1)
        gpu_load_after = gpu.load*100
        gpu_ram_after = gpu.memoryUsed
        
        self.training_time = end_time - start_time
        self.training_cpu_load_used = cpu_after - cpu_before
        self.training_ram_used = mem_after - mem_before
        self.training_gpu_load_used = gpu_load_after - gpu_load_before
        self.training_gpu_ram_used = gpu_ram_after - gpu_ram_before
        
        return fit_result
    
    def predict(self, X):
        # Create a Process object for the current process
        process = psutil.Process(os.getpid())
        
        # Before testing
        mem_before = process.memory_info().rss / (1024.0 ** 3)
        cpu_before = process.cpu_percent(interval=1)
        start_time = time.time()
        
        predict_result = self.estimator.predict(X)
        
        # After testing
        end_time = time.time()
        mem_after = process.memory_info().rss / (1024.0 ** 3)
        cpu_after = process.cpu_percent(interval=1)
        
        self.testing_time = end_time - start_time
        self.testing_ram_used = mem_after - mem_before
        self.testing_cpu_used = cpu_after - cpu_before
        
        return predict_result
    
class DecisionTreeEvaluator(ModelEvaluator):
    
    def __init__(self) -> None:
        super().__init__()
        
        self.models_collection = {}
        self.__define_models()

    def __define_models(self):
        logistic_regression_classifier = LogisticRegression(solver='newton-cg', random_state=CONFIG['GLOBAL_SEED'])
        kn_classifier = KNeighborsClassifier()
        decision_tree_classifier = DecisionTreeClassifier(random_state=CONFIG['GLOBAL_SEED'])
        gaussian_nb_classifier = GaussianNB()
        random_forest_classifier = RandomForestClassifier(random_state=CONFIG['GLOBAL_SEED'])
        gradient_boosting_classifier = GradientBoostingClassifier(random_state=CONFIG['GLOBAL_SEED'])
        xgb_classifier = XGBClassifier(random_state=CONFIG['GLOBAL_SEED'])
        lgb_classifier = LGBMClassifier(random_state=CONFIG['GLOBAL_SEED'])
        svc_classifier = SVC(decision_function_shape='ovo')
        catb_classifier = CatBoostClassifier(random_state=CONFIG['GLOBAL_SEED'], verbose=False)
        catb_classifier = CalibratedClassifierCV(catb_classifier, method='sigmoid', cv=5)
        
        self.models_collection = {
            'LogisticRegression': DecisionTreeWrapper(logistic_regression_classifier),
            'KNeighborsClassifier': DecisionTreeWrapper(kn_classifier),
            'DecisionTreeClassifier': DecisionTreeWrapper(decision_tree_classifier),
            'GaussianNB': DecisionTreeWrapper(gaussian_nb_classifier),
            'RandomForestClassifier': DecisionTreeWrapper(random_forest_classifier),
            'GradientBoostingClassifier': DecisionTreeWrapper(gradient_boosting_classifier),
            'XGBClassifier': DecisionTreeWrapper(xgb_classifier),
            'LGBMClassifier': DecisionTreeWrapper(lgb_classifier),
            'SVC': DecisionTreeWrapper(svc_classifier),
            'CatBoostClassifier': DecisionTreeWrapper(catb_classifier)
        }
    
    def __compute_metrics(self, model_name, predictions):
        mse = mean_squared_error(self.test_genetic_disorder_y, predictions)
        accuracy = accuracy_score(self.test_genetic_disorder_y, predictions)
        precision = precision_score(self.test_genetic_disorder_y, predictions, average='weighted')
        recall = recall_score(self.test_genetic_disorder_y, predictions, average='weighted')
        f1 = f1_score(self.test_genetic_disorder_y, predictions, average='weighted')

        scores = {
            'training_time (s)': self.models_collection[model_name].training_time,
            'training_cpu_load_used (%)': self.models_collection[model_name].training_cpu_load_used,
            'training_ram_used (GB)': self.models_collection[model_name].training_ram_used,
            'training_gpu_load_used (%)': self.models_collection[model_name].training_gpu_load_used,
            'training_gpu_ram_used (GB)': self.models_collection[model_name].training_gpu_ram_used,
            'testing_time (s)': self.models_collection[model_name].testing_time,
            'testing_cpu_load_used (%)': self.models_collection[model_name].testing_cpu_load_used,
            'testing_ram_used (GB)': self.models_collection[model_name].testing_ram_used,
            'testing_gpu_load_used (%)': self.models_collection[model_name].testing_gpu_load_used,
            'testing_gpu_ram_used (GB)': self.models_collection[model_name].testing_gpu_ram_used,
            'mse': mse,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        scores = {key: str(value) for key, value in scores.items()}
        csv_headers = ['Model Name'] + list(scores.keys())
        output_data = [f'{model_name}'] + list(scores.values())

        # Create a DataFrame
        df = pd.DataFrame([output_data], columns=csv_headers)

        if not os.path.isfile(CONFIG['BENCHMARK_DECISION_TREES_PATH']):
            # Write the DataFrame to the csv file
            df.to_csv(CONFIG['BENCHMARK_DECISION_TREES_PATH'], index=False)
        else:
            # Append the DataFrame to an existing CSV file
            df.to_csv(CONFIG['BENCHMARK_DECISION_TREES_PATH'], index=False, mode='a', header=False)

    def plot_failed_predictions(self, failed_predictions_per_class):
        # Plot the number of failed predictions for each class
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(failed_predictions_per_class)), failed_predictions_per_class)
        plt.title('Number of Failed Predictions for Each Class')
        plt.xlabel('Class')
        plt.ylabel('Number of Failed Predictions')
        plt.show()

    def benchmark_models(self):
        for model_name, model in self.models_collection.items():
            print('Benchmarking model:', model_name)
            model.fit(self.train_genetic_disorder_x, self.train_genetic_disorder_y)
            predictions = model.predict(self.test_genetic_disorder_x)
            self.__compute_metrics(model_name, predictions)
            
            # Calculate the number of failed predictions for each class
            failed_predictions = np.where(self.test_genetic_disorder_y != predictions, 1, 0)
            failed_predictions_per_class = np.bincount(self.test_genetic_disorder_y[failed_predictions == 1])

            self.plot_failed_predictions(failed_predictions_per_class)
    
