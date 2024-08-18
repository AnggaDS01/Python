import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report

def evaluate_model(model, test_dataset, class_names):
  """
  Evaluate the model using confusion matrix, classification report, and other metrics.
  
  Args:
  - model: Trained Keras model.
  - test_dataset: tf.data.Dataset for evaluation.
  - class_names: List of class names corresponding to the labels.
  """
  # Extract the true labels and predicted labels
  y_true = []
  y_pred = []

  for features, labels in test_dataset:
    predictions = model.predict(features, verbose=0)
    fix_label = np.squeeze(labels) * 1.0
    fix_pred = np.squeeze(np.round(predictions))

    y_pred.extend(fix_pred)
    y_true.extend(fix_label)

  y_true = np.array(y_true)
  y_pred = np.array(y_pred)

  # Compute confusion matrix
  cm = tf.math.confusion_matrix(y_true, y_pred).numpy()

  # Display confusion matrix with class names
  plt.figure(figsize=(12, 10))
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
  plt.title('Confusion Matrix')
  plt.xlabel('Predicted Label')
  plt.ylabel('True Label')
  plt.show()

  # Print classification report with class names
  print("Classification Report:")
  print(classification_report(y_true, y_pred))