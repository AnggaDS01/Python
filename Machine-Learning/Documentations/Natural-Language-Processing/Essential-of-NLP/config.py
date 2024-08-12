import tensorflow as tf
import stopwordsiso as sw
import spacy
import os

__all__ = [
    'BASE_PATH', 
    'DATASET_PATH', 
    'NUM_LABEL', 
    'MODEL_PATH', 
    'INPUT_SHAPE_MODEL_V1,'
    'INPUT_SHAPE_MODEL_V2,'
    'INPUT_SHAPE_MODEL_V3,'
    'INPUT_SHAPE_MODEL_V4,'
    'BATCH_SIZE', 
    'TRAIN_PROPORTION', 
    'OPTIMIZER', 
    'LOSS', 
    'METRICS',
    'EN_NLP',
    'EN_SW',
    'JP_NLP'
]

# Base paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
ASSETS_PATH = os.path.join(BASE_PATH, 'Assets')
DATASET_PATH = os.path.join(ASSETS_PATH, 'Datasets', 'SMSSpamCollection')
MODEL_PATH = os.path.join(ASSETS_PATH, 'Model', 'spamdetection_model_v2.keras')

# text processing settings
INPUT_SHAPE_MODEL_V1 = (3,)
INPUT_SHAPE_MODEL_V2 = (4,)
INPUT_SHAPE_MODEL_V3 = (4,)
INPUT_SHAPE_MODEL_V4 = (6,)
NUM_LABEL = 1

# Training settings
BATCH_SIZE = 32
TRAIN_PROPORTION = 0.8

# Model settings
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.001)
LOSS = tf.keras.losses.BinaryCrossentropy()
METRICS = ['accuracy']

EN_NLP = spacy.load("en_core_web_sm")
EN_SW = sw.stopwords(langs='en')
JP_NLP = spacy.load("ja_core_news_sm")