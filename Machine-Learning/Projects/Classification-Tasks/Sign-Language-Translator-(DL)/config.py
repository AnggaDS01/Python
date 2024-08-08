import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, 'Assets', 'Datasets', 'SIBI dataset')

TRAINING_PATH = os.path.join(DATASET_PATH, 'Train', '*', '*')
TEST_PATH = os.path.join(DATASET_PATH, 'Test', '*')