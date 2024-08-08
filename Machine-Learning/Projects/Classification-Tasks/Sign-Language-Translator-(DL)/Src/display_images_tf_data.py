import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

def show_multiple_images_in_tf_dataset(
        dataset, 
        num_images, 
        figsize_per_image=(2, 2),
        classes_list=None
    ):

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
            image, label = data
        else:
            image = data
            label = None
        
        image = image.numpy()
        
        # If the image is grayscale and has shape (96, 96), add an extra dimension for proper display
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        
        plt.subplot(num_rows, num_columns, idx + 1)
        plt.imshow(np.squeeze(image), cmap='gray')
        
        # If keypoints are present, plot them
        if label is not None:
            plt.title(classes_list[tf.argmax(label).numpy()])
        plt.axis('off')
    
    plt.show()