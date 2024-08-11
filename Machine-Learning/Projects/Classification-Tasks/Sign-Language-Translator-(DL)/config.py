import tensorflow as tf
import os
import glob

__all__ = [
    'BASE_PATH', 
    'DATASET_PATH', 
    'TRAINING_PATH',
    'TEST_PATH', 
    'CLASSES_LIST', 
    'MODEL_PATH', 
    'IMAGE_SIZE', 
    'INPUT_SHAPE', 
    'OFFSET_PREP_IMAGE_FOR_PREDICTION', 
    'BATCH_SIZE', 
    'TRAIN_PROPORTION', 
    'OPTIMIZER', 
    'LOSS', 
    'METRICS'
]

# Base paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
ASSETS_PATH = os.path.join(BASE_PATH, 'Assets')
DATASET_PATH = os.path.join(ASSETS_PATH, 'Datasets', 'SIBI dataset')
MODEL_PATH = os.path.join(ASSETS_PATH, 'Model', 'SIBI_effecientnetb0_model.keras')

# Dataset paths
TRAINING_PATH = os.path.join(DATASET_PATH, 'Train', '*', '*')
TEST_PATH = os.path.join(DATASET_PATH, 'Test', '*')

# Class list extraction
_class_paths = glob.glob(os.path.join(DATASET_PATH, 'Train', '*'))
CLASSES_LIST = sorted([os.path.basename(label) for label in _class_paths])

# Image processing settings
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (*IMAGE_SIZE, 3)
OFFSET_PREP_IMAGE_FOR_PREDICTION = 10

# Training settings
BATCH_SIZE = 32
TRAIN_PROPORTION = 0.9

# Model settings
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.0001)
LOSS = tf.keras.losses.CategoricalCrossentropy()
METRICS = ['accuracy']