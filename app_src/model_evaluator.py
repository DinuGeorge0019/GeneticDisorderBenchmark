
import pandas as pd
from sklearn.preprocessing import StandardScaler

from app_cfg import Config

CONFIG = Config.get_config()

class ModelEvaluator():
    def __init__(self) -> None:
        self.train_dataset_df = None
        self.train_genetic_disorder_x = None
        self.train_genetic_disorder_y = None
        
        self.test_dataset_df = None
        self.test_genetic_disorder_x = None
        self.test_genetic_disorder_y = None
        
        self._read_datasets()
        
    def _read_training_dataset(self):
        self.train_dataset_df = pd.read_csv(CONFIG['TRAIN_DATASET_PATH'])
        self.train_dataset_df = self.train_dataset_df.drop("disorder_subclass", axis=1)
        self.train_genetic_disorder_x = self.train_dataset_df.drop("genetic_disorder",axis=1)
        self.train_genetic_disorder_y = self.train_dataset_df["genetic_disorder"]
     
    def _read_testing_dataset(self):
        self.test_dataset_df = pd.read_csv(CONFIG['TEST_DATASET_PATH'])
        self.test_dataset_df = self.test_dataset_df.drop("disorder_subclass", axis=1)
        self.test_genetic_disorder_x = self.test_dataset_df.drop("genetic_disorder",axis=1)
        self.test_genetic_disorder_y = self.test_dataset_df["genetic_disorder"]
    
    def _preprocess_datasets(self):
        # Initialize the scaler
        scaler = StandardScaler()

        # Fit the scaler and transform the training data
        self.train_genetic_disorder_x = scaler.fit_transform(self.train_genetic_disorder_x)

        # Use the same scaler to transform the test data
        self.test_genetic_disorder_x = scaler.transform(self.test_genetic_disorder_x)
    
    def _read_datasets(self):
        self._read_training_dataset()
        self._read_testing_dataset()
        self._preprocess_datasets()
    