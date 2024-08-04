import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math

def show_multiple_images_with_keypoints_in_tf_datasets(
        dataset, 
        num_images, 
        figsize_per_image=(2, 2), 
        scatter_size=10, 
        scatter_color='blue'
    ):
    """
    Display multiple images with facial keypoints from a TensorFlow dataset.

    Args:
        dataset (tf.data.Dataset): TensorFlow dataset containing images and keypoints.
        num_images (int): Number of images to display.
        figsize_per_image (tuple, optional): Size of each image in the grid (width, height). Default is (2, 2).
        scatter_size (int, optional): Size of scatter plot points for keypoints. Default is 10.
        scatter_color (str, optional): Color of scatter plot points for keypoints. Default is 'blue'.

    Returns:
        None

    Example:
        >>> import tensorflow as tf
        >>> from keypoints_display_tf import show_multiple_images_with_keypoints_in_tf_datasets
        >>>
        >>> # Create a mock TensorFlow dataset for demonstration
        >>> def gen():
        >>>     for i in range(10):
        >>>         image = np.random.rand(96, 96, 1).astype(np.float32)
        >>>         keypoints = np.random.rand(10).astype(np.float32) * 96
        >>>         yield image, keypoints
        >>>         
        >>> dataset = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))
        >>> 
        >>> show_multiple_images_with_keypoints_in_tf_datasets(dataset, num_images=5)
    """
    # Shuffle and take a subset of the dataset
    dataset = dataset.shuffle(buffer_size=dataset.cardinality().numpy()).take(num_images)
    
    # Calculate the number of rows and columns needed
    num_columns = math.ceil(math.sqrt(num_images))
    num_rows = math.ceil(num_images / num_columns)
    figsize = (num_columns * figsize_per_image[0], num_rows * figsize_per_image[1])

    # Create plot
    plt.figure(figsize=figsize)

    for idx, data in enumerate(dataset):
        if len(data) == 2:
            image, keypoint = data
        else:
            image = data
            keypoint = None
        
        image = image.numpy()
        
        # If the image is grayscale and has shape (96, 96), add an extra dimension for proper display
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        
        plt.subplot(num_rows, num_columns, idx + 1)
        plt.imshow(np.squeeze(image), cmap='gray')
        
        # If keypoints are present, plot them
        if keypoint is not None:
            keypoint = keypoint.numpy()
            x_coords = keypoint[::2]
            y_coords = keypoint[1::2]
            plt.scatter(x_coords, y_coords, s=scatter_size, c=scatter_color, marker='o')
        
        plt.axis('off')
    
    plt.show()