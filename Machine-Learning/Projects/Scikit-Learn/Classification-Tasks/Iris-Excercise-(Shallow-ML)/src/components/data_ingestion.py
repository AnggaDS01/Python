import os
import sys
from pathlib import Path

config_path = Path(os.path.dirname(__file__)).parent # c:\Workspace\Python\Machine-Learning\Projects\Scikit-Learn\Classification-Tasks\Iris-Excercise-(Shallow-ML)\src
sys.path.append(str(config_path))

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from dataclasses import dataclass
import pandas as pd

from exeption import CustomExeption
from logger import logging
from data_transformation import DataTansformation, DataTransformationConfig
from model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")

        try:
            df=pd.read_csv('notebooks/datasets/IRIS.csv')
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, train_size=.8, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestian of the data is completed")

            return (
                self.ingestion_config.raw_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomExeption(e, sys)
        
if __name__ == "__main__":
    model = QuadraticDiscriminantAnalysis()
    # Kombinasi parameter yang akan diuji oleh GridSearchCV
    param_grid = {
        'reg_param': [0.0, 0.01, 0.1, 0.5, 0.9, 1.0],  # Nilai regularisasi
        'tol': [1e-4, 1e-3, 1e-2, 1e-1],              # Toleransi untuk konvergensi
        'store_covariance': [True, False],            # Apakah menyimpan covariance matrices atau tidak
    }

    obj = DataIngestion()
    data_path, train_data_path, test_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTansformation()
    train_df, test_df, _, _ = data_transformation.initiate_data_transformation(data_path, train_data_path, test_data_path)

    model_trainer = ModelTrainer()
    model_trained, model_report = model_trainer.initiate_model_trainer(train_df, test_df, model, param_grid)

    print(model_report['classification_report'])
    print(model_report['best_parameters'])
    print(model_report['best_cross_validastion_score'])