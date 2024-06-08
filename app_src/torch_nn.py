import os
import pandas as pd
import numpy as np
import psutil
import time
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, recall_score, mean_squared_error, precision_score, f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
import GPUtil

from app_cfg import Config
from .model_evaluator import ModelEvaluator

CONFIG = Config.get_config()


class PyTorchNNModel(nn.Module):
    def __init__(self, number_of_features, number_of_classes) -> None:
        super(PyTorchNNModel, self).__init__()
        self.layer1 = nn.Linear(number_of_features, 256)
        self.layer2 = nn.Linear(256, 160)
        self.layer3 = nn.Linear(160, 32)
        self.layer4 = nn.Linear(32, 128)
        self.layer5 = nn.Linear(128, 64)
        self.layer6 = nn.Linear(64, 256)
        self.layer7 = nn.Linear(256, number_of_classes)

    def forward(self, x):
        x = nn.ReLU()(self.layer1(x))
        x = nn.ReLU()(self.layer2(x))
        x = nn.ReLU()(self.layer3(x))
        x = nn.ReLU()(self.layer4(x))
        x = nn.ReLU()(self.layer5(x))
        x = nn.ReLU()(self.layer6(x))
        x = nn.Softmax(dim=1)(self.layer7(x))
        return x


class PyTorchNNModelWrapper():
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
        
        self.train_loss = []
        self.val_loss = []
        self.train_accuracy = []
        self.val_accuracy = []
        
    def parameters(self):
        return self.estimator.parameters()
    
    def validation(self, val_data, val_dataloader, criterion):
        self.estimator.eval()  # set the model to evaluation mode
        total_epoch_val_loss = 0
        correct_val_predictions = 0

        with torch.no_grad():  # disable gradient calculation
            for inputs, labels in val_dataloader:
                outputs = self.estimator(inputs)
                loss = criterion(outputs, labels)

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct_val_predictions += (predicted == labels).sum().item()

            total_epoch_val_loss += loss.cpu().item()
            epoch_val_accuracy = correct_val_predictions / len(val_data)

            print(f'Validation: loss = {total_epoch_val_loss / len(val_dataloader)}, accuracy = {epoch_val_accuracy}')
            self.val_loss.append(total_epoch_val_loss / len(val_dataloader))
            self.val_accuracy.append(epoch_val_accuracy)
            
        self.estimator.train()  # set the model back to training mode
        
    
    def fit(self, train_genetic_disorder_x, train_genetic_disorder_y, val_genetic_disorder_x, val_genetic_disorder_y, epochs, batchs, criterion, optimizer):
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
        
        train_data = TensorDataset(
            train_genetic_disorder_x, 
            train_genetic_disorder_y
        )
        
        val_data = TensorDataset(
            val_genetic_disorder_x,
            val_genetic_disorder_y
        )
        
        train_dataloader = DataLoader(train_data, batch_size=batchs, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=batchs, shuffle=True)

        for epoch in range(epochs):
            self.estimator.train()
            total_epoch_train_loss = 0
            correct_predictions = 0
            
            for inputs, labels in train_dataloader:
                optimizer.zero_grad()
                                
                outputs = self.estimator(inputs)
                loss = criterion(outputs, labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                
                loss.backward()
                optimizer.step()
                
            total_epoch_train_loss += loss.cpu().item()
            epoch_accuracy = correct_predictions / len(train_data)

            print(f'Epoch {epoch + 1}/{epochs}: train loss = {total_epoch_train_loss / len(train_dataloader)} train accuracy = {epoch_accuracy}')
            self.train_loss.append(total_epoch_train_loss / len(train_dataloader))
            self.train_accuracy.append(epoch_accuracy)
            
            self.validation(val_data, val_dataloader, criterion)  # Perform validation at the end of each epoch
                
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
        
    def plot_training_history(self, train_loss, train_accuracy, val_loss, val_accuracy):
        # summarize history for accuracy
        plt.figure(figsize=(10,5))
        plt.subplot(1, 2, 1)
        plt.plot(train_accuracy)
        plt.plot(val_accuracy)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        # summarize history for loss
        plt.subplot(1, 2, 2)
        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        plt.tight_layout()
        
        # Save the figure
        plt.savefig(CONFIG['BENCHMARK_TORCH_NN_TRAIN_GRAPH_PATH'] + f'/training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()


    def predict(self, X):
        # Create a Process object for the current process
        process = psutil.Process(os.getpid())
        
        # Before testing
        mem_before = process.memory_info().rss / (1024.0 ** 3)
        cpu_before = process.cpu_percent(interval=1)
        start_time = time.time()
        
        predict_result = self.estimator(X)
        
        # After testing
        end_time = time.time()
        mem_after = process.memory_info().rss / (1024.0 ** 3)
        cpu_after = process.cpu_percent(interval=1)
        
        self.testing_time = end_time - start_time
        self.testing_ram_used = mem_after - mem_before
        self.testing_cpu_used = cpu_after - cpu_before
        
        self.plot_training_history(self.train_loss, self.train_accuracy, self.val_loss, self.val_accuracy)
        
        return predict_result

class PyTorchNNEvaluator(ModelEvaluator):
    
    def __init__(self) -> None:
        self.val_genetic_disorder_x = None
        self.val_genetic_disorder_y = None
        self.epochs = 10
        self.batchs = 32
        
        super().__init__()

    def __create_validation_dataset(self):
        self.train_genetic_disorder_x, self.val_genetic_disorder_x, self.train_genetic_disorder_y, self.val_genetic_disorder_y = \
        train_test_split(self.train_genetic_disorder_x, self.train_genetic_disorder_y, test_size=0.1, random_state=CONFIG['GLOBAL_SEED'])
        
    def __update_to_categorical(self):
        self.train_genetic_disorder_x = torch.from_numpy(self.train_genetic_disorder_x).float()
        self.val_genetic_disorder_x = torch.from_numpy(self.val_genetic_disorder_x).float()
        self.test_genetic_disorder_x = torch.from_numpy(self.test_genetic_disorder_x).float()

        self.train_genetic_disorder_y = torch.from_numpy(self.train_genetic_disorder_y.values).long()
        self.val_genetic_disorder_y = torch.from_numpy(self.val_genetic_disorder_y.values).long()
        self.test_genetic_disorder_y = torch.from_numpy(self.test_genetic_disorder_y.values).long()
    
    def _read_datasets(self):
        self._read_training_dataset()
        self._read_testing_dataset()
        self._preprocess_datasets()
        self.__create_validation_dataset()
        self.__update_to_categorical()
    
    def __build_model(self):
        _NUM_CLASSES = 3
        model = PyTorchNNModel(self.train_genetic_disorder_x.shape[1], _NUM_CLASSES)
        return model
    
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

        if not os.path.isfile(CONFIG['BENCHMARK_TORCH_NN_PATH']):
            # Write the DataFrame to the csv file
            df.to_csv(CONFIG['BENCHMARK_TORCH_NN_PATH'], index=False)
        else:
            # Append the DataFrame to an existing CSV file
            df.to_csv(CONFIG['BENCHMARK_TORCH_NN_PATH'], index=False, mode='a', header=False)
    
    def plot_failed_predictions(self, failed_predictions_per_class):
        # Plot the number of failed predictions for each class
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(failed_predictions_per_class)), failed_predictions_per_class)
        plt.title('Number of Failed Predictions for Each Class')
        plt.xlabel('Class')
        plt.ylabel('Number of Failed Predictions')
        plt.show()

    def benchmark_model(self):
        print('Benchmarking PyTorch Neural Network')

        print('Start Training PyTorch Neural Network')
        
        model = PyTorchNNModelWrapper(self.__build_model())
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=2e-3)
        
        model.fit(
            self.train_genetic_disorder_x,
            self.train_genetic_disorder_y,
            self.val_genetic_disorder_x,
            self.val_genetic_disorder_y,
            epochs=self.epochs,
            batchs=self.batchs,
            criterion=criterion,
            optimizer=optimizer
        )

        print('Start Evaluating PyTorch Neural Network')
        test_predictions = model.predict(self.test_genetic_disorder_x)
        test_predictions = np.argmax(test_predictions.detach().numpy(), axis=1)
        self.test_genetic_disorder_y = self.test_genetic_disorder_y.cpu().numpy().squeeze()
        date_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.__compute_metrics(f'PyTorchNN_{date_time}', model, test_predictions)
        
        
        # Calculate the number of failed predictions for each class
        failed_predictions = np.where(self.test_genetic_disorder_y != test_predictions, 1, 0)
        failed_predictions_per_class = np.bincount(self.test_genetic_disorder_y[failed_predictions == 1])

        self.plot_failed_predictions(failed_predictions_per_class)
