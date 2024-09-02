import os
import sys
from pathlib import Path

config_path = Path(os.path.dirname(__file__)).parent
sys.path.append(str(config_path))

import pandas as pd

from exeption import CustomExeption
from logger import logging
from utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            features_preprocessor_path = 'artifacts/features_preprocessor.pkl'
            label_preprocessor_path = 'artifacts/label_preprocessor.pkl'

            model = load_object(file_path = model_path)
            features_preprocessor = load_object(file_path = features_preprocessor_path)
            label_preprocessor = load_object(file_path = label_preprocessor_path)

            data_transformed = features_preprocessor.transform(features)

            pred = model.predict(data_transformed)

            ord_enc = label_preprocessor.named_transformers_['label_pipeline'] \
                    .named_steps['ordinal_encoder']

            return ord_enc.inverse_transform(pred.reshape(-1,1))[0]
        
        except Exception as e:
            raise CustomExeption(e, sys)

class CustomData:
    def __init__(self,
        sepal_length: float, 
        sepal_width: float,
        petal_length: float, 
        petal_width: float
    ):

        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width

    def get_data_as_dataframe(self):
        try:
            custom_data_input = {
                'sepal_length': [self.sepal_length],
                'sepal_width': [self.sepal_width],
                'petal_length': [self.petal_length],
                'petal_width': [self.petal_width],
            }

            return pd.DataFrame(custom_data_input)
        except Exception as e:
            raise CustomExeption(e, sys)