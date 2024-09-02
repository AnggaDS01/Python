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

class DataTransformationConfig:
    features_preprocessor_obj_path = os.path.join("artifacts", "features_preprocessor.pkl")
    label_preprocessor_obj_path = os.path.join("artifacts", "label_preprocessor.pkl")

class DataTansformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, data_path):
        try:
            df = pd.read_csv(data_path, nrows=150)
            
            label_column_name = ['species']
            features_df = df.drop(columns=label_column_name)
            
            numeric_columns = features_df.select_dtypes(include=['number']).columns
            # category_columns = features_df.select_dtypes(include=['object']).columns

            num_pipeline = Pipeline(
                steps=[
                    ('min_max_scaler', MinMaxScaler()),
                ]
            )

            label_pipeline = Pipeline(
                steps=[
                    ('ordinal_encoder', OrdinalEncoder()),
                ]
            )

            logging.info(f'Numerical columns: {numeric_columns}')
            logging.info(f'label column category: {label_column_name}')

            features_preprocessor_obj = ColumnTransformer(
                [
                    ("numerical_pipeline", num_pipeline, numeric_columns),
                ]
            )

            label_preprocessor_obj = ColumnTransformer(
                [
                    ("label_pipeline", label_pipeline, label_column_name),
                ]
            )

            return features_preprocessor_obj, label_preprocessor_obj

        except Exception as e:
            raise CustomExeption(e, sys)
    
    def initiate_data_transformation(self, data_path, train_path, test_path):
        try:
            logging.info('Read train and test data completed')
            df = pd.read_csv(data_path)
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            label_column_name = ['species']

            feature_train_df = train_df.drop(columns=label_column_name)
            label_train_df = train_df[label_column_name]

            feature_test_df = test_df.drop(columns=label_column_name)
            label_test_df = test_df[label_column_name]

            logging.info('Obtaining preprocessing object')
            features_preprocessor_obj, label_preprocessor_obj = self.get_data_transformer_object(data_path)

            logging.info(f'Applying preprocessing object on training dataframe and testing dataframe')
            features_preprocessor_obj.fit(df)
            label_preprocessor_obj.fit(df)

            feature_train_df[feature_train_df.columns] = features_preprocessor_obj.transform(feature_train_df)
            feature_test_df[feature_test_df.columns] = features_preprocessor_obj.transform(feature_test_df)

            label_train_df = pd.DataFrame(label_preprocessor_obj.transform(label_train_df), columns=label_train_df.columns)
            label_test_df = pd.DataFrame(label_preprocessor_obj.transform(label_test_df), columns=label_test_df.columns)

            train_df_transformed = pd.concat([feature_train_df.reset_index(drop=True), label_train_df], axis=1)
            test_df_transformed = pd.concat([feature_test_df.reset_index(drop=True), label_test_df], axis=1)
            
            logging.info('Saved preprocessing object.')
            save_object(
                file_path=self.data_transformation_config.features_preprocessor_obj_path,
                obj=features_preprocessor_obj
            )

            save_object(
                file_path=self.data_transformation_config.label_preprocessor_obj_path,
                obj=label_preprocessor_obj
            )

            return (
                train_df_transformed,
                test_df_transformed,
                self.data_transformation_config.features_preprocessor_obj_path,
                self.data_transformation_config.label_preprocessor_obj_path,
            )

        except Exception as e:
            raise CustomExeption(e, sys)
        











# ==================== Uncomment Jika Kasusnya Regresi ====================
# df = pd.read_csv(data_path)
# train_df = pd.read_csv(train_path)
# test_df = pd.read_csv(test_path)

# label_column_name = 'species'

# feature_train_df = train_df.drop(columns=label_column_name)
# label_train_df = train_df[label_column_name]

# feature_test_df = test_df.drop(columns=label_column_name)
# label_test_df = test_df[label_column_name]

# preprocessing_obj = self.get_data_transformer_object(data_path)
# preprocessing_obj.fit(df)

# feature_train_df[feature_train_df.columns] = preprocessing_obj.transform(feature_train_df)
# feature_test_df[feature_test_df.columns] = preprocessing_obj.transform(feature_test_df)

# train_df_transformed = pd.concat([feature_train_df, label_train_df], axis=1)
# test_df_transformed = pd.concat([feature_test_df, label_test_df], axis=1)
# ==================== Uncomment Jika Kasusnya Regresi ====================