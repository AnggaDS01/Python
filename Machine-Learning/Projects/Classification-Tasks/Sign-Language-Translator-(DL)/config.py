import tensorflow as tf
import os
import glob

__all__ = [
    'BASE_PATH', 
    'DATASET_PATH', 
    'TEST_PATH', 
    'CLASSES_LIST', 
    'MODEL_PATH', 
    'INPUT_SHAPE', 
    'IMAGE_SIZE', 
    'OFFSET_PREP_IMAGE_FOR_PREDDICTION'
]

BASE_PATH=os.path.dirname(os.path.abspath(__file__))
DATASET_PATH=os.path.join(BASE_PATH, 'Assets', 'Datasets', 'SIBI dataset')

TRAINING_PATH=os.path.join(DATASET_PATH, 'Train', '*', '*')
TEST_PATH=os.path.join(DATASET_PATH, 'Test', '*')

_class_paths=glob.glob(os.path.join(DATASET_PATH, 'Train', '*'))

# Mengambil nama folder sebagai kelas
CLASSES_LIST=[os.path.basename(label) for label in _class_paths]
CLASSES_LIST.sort()

MODEL_PATH=os.path.join(BASE_PATH, 'Assets', 'Model', 'SIBI_effecientnetb0_model.keras')

IMAGE_SIZE=(224, 224)
INPUT_SHAPE=(224, 224, 3)
OFFSET_PREP_IMAGE_FOR_PREDDICTION=10

BATCH_SIZE=32
TRAIN_PROPORTION=0.9
OPTIMIZER=tf.keras.optimizers.Adam(learning_rate=0.0001)
LOSS=tf.keras.losses.CategoricalCrossentropy()
METRICS=['accuracy']

### MINTA BENERIN CHAT GPT, APAKAH PERLU ADA TAMBAHAN, DAN JUGA SUSUNANNYA DI PERBAIKI
### PERBAIKI BAGIAN PREDICT GESTURE OPENCV AGAR LEBIH MODULAR