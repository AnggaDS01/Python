import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

class Visualizer:
    def __init__(self, figsize_per_image=(2, 2)):
        """
        Initializes the Visualizer with the size of each image in the grid.

        Args:
            figsize_per_image (tuple): Size (width, height) of each image in inches.
        """
        self.figsize_per_image = figsize_per_image

    def show_multiple_images_in_tf_data(self, dataset, num_images, classes_list=None):
        """
        Display multiple images from a TensorFlow dataset.

        Args:
            dataset (tf.data.Dataset): TensorFlow dataset containing images.
            num_images (int): Number of images to display.
            classes_list (list, optional): List of class names for labeling. Defaults to None.

        Returns:
            None: Displays the images using matplotlib.

        Example:
            >>> import tensorflow as tf
            >>> from your_module import Visualizer
            >>> dataset = tf.data.Dataset.from_tensor_slices(images)  # Replace with actual dataset
            >>> visualizer = Visualizer(figsize_per_image=(4, 4))
            >>> visualizer.show_multiple_images_in_tf_data(dataset, num_images=9, classes_list=classes_list)
        """
        # Shuffle and take a subset of the dataset
        dataset = dataset.shuffle(buffer_size=dataset.cardinality().numpy()).take(num_images)
        
        # Calculate the number of rows and columns needed
        num_columns = math.ceil(math.sqrt(num_images))
        num_rows = math.ceil(num_images / num_columns)
        figsize = (num_columns * self.figsize_per_image[0], num_rows * self.figsize_per_image[1])

        # Create plot
        plt.figure(figsize=figsize)

        for idx, data in enumerate(dataset):
            if len(data) == 2:
                image, label = data
            else:
                image = data
                label = None
            
            image = image.numpy()
            
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)
            
            plt.subplot(num_rows, num_columns, idx + 1)
            plt.imshow(np.squeeze(image), cmap='gray')
            
            # If keypoints are present, plot them
            if label is not None:
                plt.title(classes_list[tf.argmax(label).numpy()])
            plt.axis('off')
        
        plt.show()

    def show_multiple_images_with_label_pred_tf_data(self, model, dataset, num_images, classes_list=None):
        """
        Display multiple images from a TensorFlow dataset with predicted labels.

        Args:
            model (tf.keras.Model): Trained TensorFlow/Keras model for predictions.
            dataset (tf.data.Dataset): TensorFlow dataset containing images.
            num_images (int): Number of images to display.
            classes_list (list, optional): List of class names for labeling. Defaults to None.

        Returns:
            None: Displays the images with predicted labels using matplotlib.

        Example:
            >>> import tensorflow as tf
            >>> from your_module import Visualizer
            >>> model = tf.keras.models.load_model('path_to_model')
            >>> dataset = tf.data.Dataset.from_tensor_slices(images)  # Replace with actual dataset
            >>> visualizer = Visualizer(figsize_per_image=(4, 4))
            >>> visualizer.show_multiple_images_with_label_pred_tf_data(model, dataset, num_images=9, classes_list=classes_list)
        """
        # Shuffle and take a subset of the dataset
        dataset = dataset.shuffle(buffer_size=dataset.cardinality().numpy()).take(num_images)

        # Calculate the number of rows and columns needed
        num_columns = math.ceil(math.sqrt(num_images))
        num_rows = math.ceil(num_images / num_columns)
        figsize = (num_columns * self.figsize_per_image[0], num_rows * self.figsize_per_image[1])

        # Create plot
        plt.figure(figsize=figsize)

        for idx, image in enumerate(dataset):
            image = tf.expand_dims(image, axis=0)
            label = model.predict(image, verbose=0)
            label = tf.squeeze(label)
            alphabet_label = classes_list[tf.argmax(label).numpy()]

            plt.subplot(num_rows, num_columns, idx + 1)
            plt.imshow(np.squeeze(image), cmap='gray')
            plt.title(alphabet_label)
            plt.axis('off')

        plt.show()
