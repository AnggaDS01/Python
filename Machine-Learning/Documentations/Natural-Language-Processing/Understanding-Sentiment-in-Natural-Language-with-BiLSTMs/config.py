import tensorflow as tf
import os
import glob

__all__ = [
    'BASE_PATH', 
    'DATASET_PATH', 
    'LSTM_MODEL_PATH', 
    'BI_LSTM_MODEL_PATH', 
    'BATCH_SIZE', 
    'TRAIN_PROPORTION', 
    'VOCAB_SIZE',
    'OOV_TOKEN',
    'MAX_PAD',
    'EMBEDDING_DIM',
    'OPTIMIZER', 
    'LOSS', 
    'METRICS',
]

# Base paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
ASSETS_PATH = os.path.join(BASE_PATH, 'Assets')
DATASET_PATH = os.path.join(ASSETS_PATH, 'Datasets', 'IMDB Dataset.csv')

LSTM_MODEL_PATH = os.path.join(ASSETS_PATH, 'Model', 'imdb_sentiment_review_bilstm.keras')
BI_LSTM_MODEL_PATH = os.path.join(ASSETS_PATH, 'Model', 'imdb_sentiment_review_bilstm.keras')

# Training settings
BATCH_SIZE = 64
TRAIN_PROPORTION = 0.8

# Tokenizer settings
VOCAB_SIZE=10000 
OOV_TOKEN="<OOV>" 
MAX_PAD=235
EMBEDDING_DIM = 64

# Model settings
OPTIMIZER = tf.keras.optimizers.Adam()
LOSS = tf.keras.losses.BinaryCrossentropy()
METRICS = ['accuracy', 'precision', 'recall']