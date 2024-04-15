

# standard library imports
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# local library specific imports
from app_cfg import Config

# get global configuration
CONFIG = Config.get_config()

class DatasetBuilder():
    
    def __init__(self) -> None:
        self.raw_dataset_df = self.__read_raw_dataset()
    
    
    def __read_raw_dataset(self):
        dataset_df = pd.read_csv(CONFIG['RAW_DATASET_PATH'])
        return dataset_df
    
    
    def __remove_not_needed_columns(self):
        columns_to_remove = ['Patient Id', 'Patient First Name', 'Family Name', "Father's name", 'Institute Name', 
                     'Location of Institute', 'Parental consent', 'Follow-up', 'Status']
        self.raw_dataset_df = self.raw_dataset_df.drop(columns=columns_to_remove)
    
    
    def __rename_columns(self):
        self.raw_dataset_df.rename(columns={"Patient Age":"patient_age",
                    "Genes in mother's side":"mother_side_genes",
                    'Inherited from father':"father_side_genes",
                    'Maternal gene':"maternal_gene",
                    'Paternal gene':"paternal_gene",
                    "Blood cell count (mcL)":"blood_cell_count",
                    "Mother's age":"mother_age",
                    "Father's age":"father_age",
                    "Respiratory Rate (breaths/min)":"respiratory_rate",
                    "Heart Rate (rates/min":"heart_rate",
                    'Gender': 'gender',
                    "Birth asphyxia":"birth_asphyxia",
                    "Autopsy shows birth defect (if applicable)":"authopsy_birth_defect",
                    "Place of birth":"place_of_birth",
                    "Folic acid details (peri-conceptional)":"follic_acid",
                    "H/O serious maternal illness":"maternal_illness",
                    "H/O radiation exposure (x-ray)":"radiation_exposure",
                    "H/O substance abuse":"substance_abuse",
                    "Assisted conception IVF/ART":"assist_conception",
                    "History of anomalies in previous pregnancies":"previous_pregnancies_anomalies",
                    "No. of previous abortion":"previous_abortion",
                    "Birth defects":"birth_defects",
                    "White Blood cell count (thousand per microliter)":"white_blood_cell_count",
                    "Blood test result":"blood_test_result",
                    "Genetic Disorder":"genetic_disorder",
                    "Disorder Subclass":"disorder_subclass",
                    'Test 1':'test_1',
                    'Test 2':'test_2',
                    'Test 3':'test_3',
                    'Test 4':'test_4',
                    'Test 5':'test_5',
                    'Symptom 1':'symptom_1',
                    'Symptom 2':'symptom_2',
                    'Symptom 3':'symptom_3',
                    'Symptom 4':'symptom_4',
                    'Symptom 5':'symptom_5'
                    },inplace=True)
    
    
    def __replace_nan_data(self):
        self.raw_dataset_df["birth_asphyxia"] = self.raw_dataset_df["birth_asphyxia"].replace("No record", np.NaN)
        self.raw_dataset_df["birth_asphyxia"] = self.raw_dataset_df["birth_asphyxia"].replace("Not available", np.NaN)

        self.raw_dataset_df["authopsy_birth_defect"] = self.raw_dataset_df["authopsy_birth_defect"].replace("None",np.NaN)
        self.raw_dataset_df["authopsy_birth_defect"] = self.raw_dataset_df["authopsy_birth_defect"].replace("Not applicable",np.NaN)

        self.raw_dataset_df["radiation_exposure"] = self.raw_dataset_df["radiation_exposure"].replace("Not applicable",np.NaN)
        self.raw_dataset_df["radiation_exposure"] = self.raw_dataset_df["radiation_exposure"].replace("-",np.NaN)

        self.raw_dataset_df["substance_abuse"] = self.raw_dataset_df["substance_abuse"].replace("Not applicable",np.NaN)
        self.raw_dataset_df["substance_abuse"] = self.raw_dataset_df["substance_abuse"].replace("-",np.NaN)

        self.raw_dataset_df.dropna(subset=['disorder_subclass', 'genetic_disorder'], inplace=True)
        
        self.raw_dataset_df["father_side_genes"].fillna(self.raw_dataset_df["father_side_genes"].mode()[0], inplace=True)
        self.raw_dataset_df["maternal_gene"].fillna(self.raw_dataset_df["maternal_gene"].mode()[0], inplace=True)
        self.raw_dataset_df["respiratory_rate"].fillna(self.raw_dataset_df["respiratory_rate"].mode()[0], inplace=True)
        self.raw_dataset_df["heart_rate"].fillna(self.raw_dataset_df["heart_rate"].mode()[0], inplace=True)
        self.raw_dataset_df["gender"].fillna(self.raw_dataset_df["gender"].mode()[0], inplace=True)
        self.raw_dataset_df["birth_asphyxia"].fillna(self.raw_dataset_df["birth_asphyxia"].mode()[0], inplace=True)
        self.raw_dataset_df["authopsy_birth_defect"].fillna(self.raw_dataset_df["authopsy_birth_defect"].mode()[0], inplace=True)
        self.raw_dataset_df["place_of_birth"].fillna(self.raw_dataset_df["place_of_birth"].mode()[0], inplace=True)
        self.raw_dataset_df["follic_acid"].fillna(self.raw_dataset_df["follic_acid"].mode()[0], inplace=True)
        self.raw_dataset_df["maternal_illness"].fillna(self.raw_dataset_df["maternal_illness"].mode()[0], inplace=True)
        self.raw_dataset_df["radiation_exposure"].fillna(self.raw_dataset_df["radiation_exposure"].mode()[0], inplace=True)
        self.raw_dataset_df["substance_abuse"].fillna(self.raw_dataset_df["substance_abuse"].mode()[0], inplace=True)
        self.raw_dataset_df["assist_conception"].fillna(self.raw_dataset_df["assist_conception"].mode()[0], inplace=True)
        self.raw_dataset_df["previous_pregnancies_anomalies"].fillna(self.raw_dataset_df["previous_pregnancies_anomalies"].mode()[0], inplace=True)
        self.raw_dataset_df["birth_defects"].fillna(self.raw_dataset_df["birth_defects"].mode()[0], inplace=True)
        self.raw_dataset_df["blood_test_result"].fillna(self.raw_dataset_df["blood_test_result"].mode()[0], inplace=True)
        self.raw_dataset_df["test_1"].fillna(self.raw_dataset_df["test_1"].mode()[0], inplace=True)
        self.raw_dataset_df["test_2"].fillna(self.raw_dataset_df["test_2"].mode()[0], inplace=True)
        self.raw_dataset_df["test_3"].fillna(self.raw_dataset_df["test_3"].mode()[0], inplace=True)
        self.raw_dataset_df["test_4"].fillna(self.raw_dataset_df["test_4"].mode()[0], inplace=True)
        self.raw_dataset_df["test_5"].fillna(self.raw_dataset_df["test_5"].mode()[0], inplace=True)
        self.raw_dataset_df["symptom_1"].fillna(self.raw_dataset_df["symptom_1"].mode()[0], inplace=True)
        self.raw_dataset_df["symptom_2"].fillna(self.raw_dataset_df["symptom_2"].mode()[0], inplace=True)
        self.raw_dataset_df["symptom_3"].fillna(self.raw_dataset_df["symptom_3"].mode()[0], inplace=True)
        self.raw_dataset_df["symptom_4"].fillna(self.raw_dataset_df["symptom_4"].mode()[0], inplace=True)
        self.raw_dataset_df["symptom_5"].fillna(self.raw_dataset_df["symptom_5"].mode()[0], inplace=True)

        self.raw_dataset_df["patient_age"].fillna(self.raw_dataset_df.groupby(["disorder_subclass"])["patient_age"].transform("mean"),inplace=True)
        self.raw_dataset_df["mother_age"].fillna(self.raw_dataset_df.groupby(["disorder_subclass"])["mother_age"].transform("mean"),inplace=True)
        self.raw_dataset_df["father_age"].fillna(self.raw_dataset_df.groupby(["disorder_subclass"])["father_age"].transform("mean"),inplace=True)
        self.raw_dataset_df["previous_abortion"].fillna(self.raw_dataset_df.groupby(["disorder_subclass"])["previous_abortion"].transform("mean"),inplace=True)
        self.raw_dataset_df["white_blood_cell_count"].fillna(self.raw_dataset_df.groupby(["disorder_subclass"])["white_blood_cell_count"].transform("mean"),inplace=True)
    
    
    def __encode_categorical_data(self):
        # Create an instance of LabelEncoder
        encoder = LabelEncoder()

        # Encode string columns
        string_columns = self.raw_dataset_df.select_dtypes(include=['object']).columns
        self.raw_dataset_df[string_columns] = self.raw_dataset_df[string_columns].apply(encoder.fit_transform)

        # Transform all the columns into float for normalization
        self.raw_dataset_df = self.raw_dataset_df.astype('float32')
    
    
    def __preprocess_dataset(self):
        self.__remove_not_needed_columns()
        self.__rename_columns()
        self.__replace_nan_data()
        self.__encode_categorical_data()
    
    
    def __create_data_analysis(self):
        genetic_disorder_distribution = self.raw_dataset_df['genetic_disorder'].value_counts()
        disorder_subclass_distribution = self.raw_dataset_df['disorder_subclass'].value_counts()
            
        logging.basicConfig(filename=CONFIG['DATASET_LOG_PATH'], level=logging.INFO)
        
        logging.info("Genetic Disorder Class Distribution:")        
        logging.info(genetic_disorder_distribution)
        logging.info("\nDisorder Subclass Distribution:")
        logging.info(disorder_subclass_distribution)

        # Plot Genetic Disorder Class Distribution
        plt.figure(figsize=(10, 5))
        genetic_disorder_distribution.plot(kind='bar')
        plt.title('Genetic Disorder Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        
        # Save the plot to a file
        plt.savefig(CONFIG['DATASET_DISTRIBUTION_PLOT_PATH'])
 
            
    def __save_dataset(self):
        # Split the data into train and test sets
        train_df, test_df = train_test_split(self.raw_dataset_df, test_size=0.2, random_state=CONFIG['GLOBAL_SEED'])
        
        train_df.to_csv(CONFIG['TRAIN_DATASET_PATH'], index=False)
        test_df.to_csv(CONFIG['TEST_DATASET_PATH'], index=False)


    def create_dataset(self):
        self.__preprocess_dataset()
        self.__create_data_analysis()
        self.__save_dataset()

