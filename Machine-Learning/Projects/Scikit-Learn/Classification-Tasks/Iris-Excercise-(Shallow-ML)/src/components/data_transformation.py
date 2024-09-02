from dataclasses import dataclass
from pathlib import Path
import os
import sys

config_path = Path(os.path.dirname(__file__)).parent # c:\Workspace\Python\Machine-Learning\Projects\Scikit-Learn\Classification-Tasks\Iris-Excercise-(Shallow-ML)\src
sys.path.append(str(config_path))
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from exeption import CustomExeption
from logger import logging
from utils import save_object

import pandas as pd
import numpy as np

class DataTransformationConfig:
    preprocessor_obj_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTansformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, data_path):
        try:
            df = pd.read_csv(data_path)
            
            target_column_name = ['species']
            features_df = df.drop(columns=target_column_name)
            
            numeric_columns = features_df.select_dtypes(include=['number']).columns
            # category_columns = features_df.select_dtypes(include=['object']).columns

            num_pipeline = Pipeline(
                steps=[
                    ('min_max_scaler', MinMaxScaler()),
                ]
            )

            target_pipeline = Pipeline(
                steps=[
                    ('ordinal_encoder', OrdinalEncoder()),
                ]
            )

            logging.info(f'Numerical columns: {numeric_columns}')
            logging.info(f'Target column category: {target_column_name}')

            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", num_pipeline, numeric_columns),
                    ("target_pipeline", target_pipeline, target_column_name),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomExeption(e, sys)
    
    def initiate_data_transformation(self, data_path, train_path, test_path):
        try:
            df = pd.read_csv(data_path)
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_object(data_path)

            logging.info(f'Applying preprocessing object on training dataframe and testing dataframe')

            preprocessing_obj.fit(df)
            categorical_target_name_list = preprocessing_obj.named_transformers_['target_pipeline'] \
                                        .named_steps['ordinal_encoder'] \
                                        .categories_[0]

            train_df[train_df.columns] = preprocessing_obj.transform(train_df)
            test_df[test_df.columns] = preprocessing_obj.transform(test_df)

            logging.info('Saved preprocessing object.')

            # ==================== Uncomment Jika Kasusnya Regresi ====================
            # df = pd.read_csv(data_path)
            # train_df = pd.read_csv(train_path)
            # test_df = pd.read_csv(test_path)

            # target_column_name = 'species'
            
            # feature_train_df = train_df.drop(columns=target_column_name)
            # target_train_df = train_df[target_column_name]

            # feature_test_df = test_df.drop(columns=target_column_name)
            # target_test_df = test_df[target_column_name]

            # preprocessing_obj = self.get_data_transformer_object(data_path)
            # preprocessing_obj.fit(df)

            # feature_train_df[feature_train_df.columns] = preprocessing_obj.transform(feature_train_df)
            # feature_test_df[feature_test_df.columns] = preprocessing_obj.transform(feature_test_df)

            # train_df_transformed = pd.concat([feature_train_df, target_train_df], axis=1)
            # test_df_transformed = pd.concat([feature_test_df, target_test_df], axis=1)
            # ==================== Uncomment Jika Kasusnya Regresi ====================

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessing_obj
            )

            return (
                train_df,
                test_df,
                categorical_target_name_list,
                self.data_transformation_config.preprocessor_obj_path
            )

        except Exception as e:
            raise CustomExeption(e, sys)