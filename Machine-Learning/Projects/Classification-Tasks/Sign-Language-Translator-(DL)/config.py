import os
import glob

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, 'Assets', 'Datasets', 'SIBI dataset')

TRAINING_PATH = os.path.join(DATASET_PATH, 'Train', '*', '*')
TEST_PATH = os.path.join(DATASET_PATH, 'Test', '*')

class_paths = glob.glob(os.path.join(DATASET_PATH, 'Train', '*'))

# Mengambil nama folder sebagai kelas
CLASSES_LIST = [os.path.basename(label) for label in class_paths]
CLASSES_LIST.sort()