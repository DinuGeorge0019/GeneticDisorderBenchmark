

import os
import pandas as pd
import numpy as np
import psutil
import time
from datetime import datetime
import GPUtil

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, recall_score, mean_squared_error, precision_score, f1_score
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras_tuner.tuners import RandomSearch
from scipy.special import softmax

from app_cfg import Config
from .model_evaluator import ModelEvaluator

CONFIG = Config.get_config()


class KerassNNModelWrapper():
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
        gpus = GPUtil.getGPUs()
        gpu = gpus[0]
        
        # Before testing
        mem_before = process.memory_info().rss / (1024.0 ** 3)
        cpu_before = process.cpu_percent(interval=1)
        gpu_load_before = gpu.load*100
        gpu_ram_before = gpu.memoryUsed
        start_time = time.time()
        
        predict_result = self.estimator.predict(X)
        
        # After testing
        end_time = time.time()
        mem_after = process.memory_info().rss / (1024.0 ** 3)
        cpu_after = process.cpu_percent(interval=1)
        gpu_load_after = gpu.load*100
        gpu_ram_after = gpu.memoryUsed
        
        self.testing_time = end_time - start_time
        self.testing_cpu_load_used = cpu_after - cpu_before
        self.testing_ram_used = mem_after - mem_before
        self.testing_gpu_load_used = gpu_load_after - gpu_load_before
        self.testing_gpu_ram_used = gpu_ram_after - gpu_ram_before
        
        return predict_result

class KerasNNEvaluator(ModelEvaluator):
    
    def __init__(self) -> None:
        self.val_genetic_disorder_x = None
        self.val_genetic_disorder_y = None
        self.epochs = 10
        
        super().__init__()

    def __create_validation_dataset(self):
        self.train_genetic_disorder_x, self.val_genetic_disorder_x, self.train_genetic_disorder_y, self.val_genetic_disorder_y = \
        train_test_split(self.train_genetic_disorder_x, self.train_genetic_disorder_y, test_size=0.1, random_state=CONFIG['GLOBAL_SEED'])
        
    def __update_to_categorical_y(self):
        self.train_genetic_disorder_y = to_categorical(self.train_genetic_disorder_y)
        self.val_genetic_disorder_y = to_categorical(self.val_genetic_disorder_y)
        self.test_genetic_disorder_y = to_categorical(self.test_genetic_disorder_y)
    
    def _read_datasets(self):
        self._read_training_dataset()
        self._read_testing_dataset()
        self._preprocess_datasets()
        self.__create_validation_dataset()
        self.__update_to_categorical_y()
    
    def __build_model(self, hp):
        _NUM_CLASSES = 3
        
        model = Sequential()
        model.add(Dense(units=self.train_genetic_disorder_x.shape[1], activation='relu'))  # First layer with number of neurons equal to number of input features
        for i in range(hp.Int('num_layers', 1, 20)):
            model.add(Dense(units=hp.Int('units_' + str(i),
                                        min_value=32,
                                        max_value=256,
                                        step=32),
                            activation='relu'))
        model.add(Dense(_NUM_CLASSES, activation='softmax'))  # _NUM_CLASSES is the number of classes
        model.compile(
            optimizer=Adam(hp.Choice('learning_rate', [2e-2, 2e-3, 2e-4])), # , 2e-2, 2e-3, 2e-4
            loss='categorical_crossentropy',  # or 'sparse_categorical_crossentropy'
            metrics=['accuracy'])
        return model
    
    def __search_best_model(self):
        # Define a tuner
        tuner = RandomSearch(
            self.__build_model,
            objective='val_accuracy',
            max_trials=5,
            executions_per_trial=3,
            directory='nn_models',
            project_name='genetic_disorder',
            seed=CONFIG['GLOBAL_SEED']
        )

        # Perform hyperparameter search
        tuner.search(
            self.train_genetic_disorder_x, self.train_genetic_disorder_y,
            epochs=self.epochs,
            validation_data=(self.val_genetic_disorder_x, self.val_genetic_disorder_y),
        )

        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

        return tuner, best_hps
    
    def __compute_metrics(self, model_name, model, predictions):
        
        mse = mean_squared_error(self.test_genetic_disorder_y, predictions)
        accuracy = accuracy_score(self.test_genetic_disorder_y, predictions)
        precision = precision_score(self.test_genetic_disorder_y, predictions, average='weighted')
        recall = recall_score(self.test_genetic_disorder_y, predictions, average='weighted')
        f1 = f1_score(self.test_genetic_disorder_y, predictions, average='weighted')

        scores = {
            'training_time (s)': model.training_time,
            'training_cpu_load_used (%)': model.training_cpu_load_used,
            'training_ram_used (GB)': model.training_ram_used,
            'training_gpu_load_used (%)': model.training_gpu_load_used,
            'training_gpu_ram_used (GB)': model.training_gpu_ram_used,
            'testing_time (s)': model.testing_time,
            'testing_cpu_load_used (%)': model.testing_cpu_load_used,
            'testing_ram_used (GB)': model.testing_ram_used,
            'testing_gpu_load_used (%)': model.testing_gpu_load_used,
            'testing_gpu_ram_used (GB)': model.testing_gpu_ram_used,
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

        if not os.path.isfile(CONFIG['BENCHMARK_KERAS_NN_PATH']):
            # Write the DataFrame to the csv file
            df.to_csv(CONFIG['BENCHMARK_KERAS_NN_PATH'], index=False)
        else:
            # Append the DataFrame to an existing CSV file
            df.to_csv(CONFIG['BENCHMARK_KERAS_NN_PATH'], index=False, mode='a', header=False)

    def plot_training_history(self, history):
        # summarize history for accuracy
        plt.figure(figsize=(10,5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        # summarize history for loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        # Save the figure
        plt.savefig(CONFIG['BENCHMARK_KERAS_NN_TRAIN_GRAPH_PATH'] + f'/training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()

    
    def benchmark_model(self):
        print('Benchmarking Keras Neural Network')
        
        print('Start Searching best parameters')
        tuner, best_hps = self.__search_best_model()
        
        print('Start Training Keras Neural Network')
        # Build the model with the optimal hyperparameters and train it on the data
        model = KerassNNModelWrapper(tuner.hypermodel.build(best_hps))

        history = model.fit(
            self.train_genetic_disorder_x, 
            self.train_genetic_disorder_y, 
            epochs=self.epochs, 
            validation_data=(
                self.val_genetic_disorder_x, 
                self.val_genetic_disorder_y
            )
        )
        
        self.plot_training_history(history)
        
        print('Start Evaluating Keras Neural Network')
        # Make predictions on the testing dataset
        test_predictions = model.predict(self.test_genetic_disorder_x)
        # Convert the predictions to class labels
        test_predictions = np.argmax(test_predictions, axis=1)
        self.test_genetic_disorder_y = np.argmax(self.test_genetic_disorder_y, axis=1)
        
        date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.__compute_metrics(f'KerasNN_{date_time}', model, test_predictions)
