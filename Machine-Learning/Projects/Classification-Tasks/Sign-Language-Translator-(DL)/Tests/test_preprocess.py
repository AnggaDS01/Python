import sys
from pathlib import Path
config_path = Path('./Machine-Learning/Projects/Classification-Tasks/Sign-Language-Translator-(DL)')
sys.path.append(str(config_path))

from Src.preprocess_tf_data import PreprocessingPipeline
from config import IMAGE_SIZE

import tensorflow as tf
import cv2

prep_pipeline_func = PreprocessingPipeline(target_size=IMAGE_SIZE, is_gray=False)

# Define a function to predict the label of an input image
def predict_gesture(model, classes_list, input_image, image_size_for_model):
  # Rescale the image to values between 0 and 1
  img_rescale = (input_image * 1.0) / 255.0
  # Resize the image to match the model input size
  img_resize = cv2.resize(img_rescale, image_size_for_model)
  # preprocess image to match the input data model
  image_preprcess = prep_pipeline_func.processing_data_test_to_VGG_input(img_resize)
  # Add a dimension to the image for batch processing
  img_resize_expand = tf.expand_dims(image_preprcess, axis=0)
  # Predict the label using the model
  alpha_prediction = model.predict(img_resize_expand, verbose=0)
  # squeeze label dim become (26, ) not (1, 26)
  alpha_prediction = tf.squeeze(alpha_prediction)
  # Get the category with the highest probability
  alpha_label = classes_list[tf.argmax(alpha_prediction).numpy()]
  # Return the predicted label
  return alpha_label