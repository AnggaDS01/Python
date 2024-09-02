from dataclasses import dataclass
from pathlib import Path
import os
import sys

config_path = Path(os.path.dirname(__file__)).parent 
sys.path.append(str(config_path))

# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from exeption import CustomExeption
from logger import logging
from utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_df, test_df, model, hyper_params):
        try:
            logging.info('Split training and test input data')
            target_column_name = 'species'

            X_train = train_df.drop(columns=target_column_name)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=target_column_name)
            y_test = test_df[target_column_name]

            # model = QuadraticDiscriminantAnalysis()

            logging.info(f"Evaluating model: {model.__class__.__name__}")
            model_trained, model_report = evaluate_model(
                X_train,
                y_train,
                X_test,
                y_test,
                model,
                hyper_params
            )

            if model_report['accuracy'] < .8:
                raise CustomExeption(f"Best Model Not Found, Model Accuracy Reached: {model_report['accuracy']}")
            
            # Tambahkan logging untuk menampilkan hasil evaluasi
            logging.info(f"Model accuracy: {model_report['accuracy']:.4f}")
            logging.info(f"Model precision: {model_report['precision']:.4f}")
            logging.info(f"Model recall: {model_report['recall']:.4f}")

            logging.info('Model evaluation completed successfully')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model_trained
            )

            return model_trained, model_report

        except Exception as e:
            raise CustomExeption(e, sys)
            