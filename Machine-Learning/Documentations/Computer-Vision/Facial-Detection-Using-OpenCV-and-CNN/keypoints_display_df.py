import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import re

def show_multiple_images_with_keypoints_in_dataframe(
        dataframe, 
        keypoints_columns,
        image_column, 
        parent_path=None,
        num_images=5, 
        figsize_per_image=(8, 4), 
        image_size=(96, 96),
        scatter_size=400,
        scatter_color='blue',
        n_seed=42
    ):
    """
    Display multiple images with facial keypoints from a Pandas dataframe.

    Args:
        dataframe (pd.DataFrame): Dataframe containing images and keypoints.
        keypoints_columns (list): List of keypoints columns.
        image_column (str): Column name containing image data or image paths.
        parent_path (str, optional): Parent path for image files if image_column contains file paths. Default is None.
        num_images (int, optional): Number of images to display. Default is 5.
        figsize_per_image (tuple, optional): Size of each image in the grid (width, height). Default is (8, 4).
        image_size (tuple, optional): Size of each image (height, width). Default is (96, 96).
        scatter_size (int, optional): Size of scatter plot points for keypoints. Default is 400.
        scatter_color (str, optional): Color of scatter plot points for keypoints. Default is 'blue'.

    Returns:
        None

    Example:
        >>> import pandas as pd
        >>> from keypoints_display_df import show_multiple_images_with_keypoints_in_dataframe
        >>> 
        >>> # Create a mock dataframe for demonstration
        >>> data = {
        >>>     'Image': ['image1.png', 'image2.png', 'image3.png'],
        >>>     'left_eye_center_x': [30, 35, 40],
        >>>     'left_eye_center_y': [50, 55, 60],
        >>>     'right_eye_center_x': [70, 75, 80],
        >>>     'right_eye_center_y': [50, 55, 60],
        >>> }
        >>> df = pd.DataFrame(data)
        >>> keypoints_columns = ['left_eye_center_x', 'left_eye_center_y', 'right_eye_center_x', 'right_eye_center_y']
        >>> 
        >>> show_multiple_images_with_keypoints_in_dataframe(df, keypoints_columns, 'Image', parent_path='/path/to/images/', num_images=3)
    """
    # Calculate the number of rows and columns needed
    num_columns = math.ceil(math.sqrt(num_images))
    num_rows = math.ceil(num_images / num_columns)
    figsize = (num_columns * figsize_per_image[0], num_rows * figsize_per_image[1])

    # Create plot
    plt.figure(figsize=figsize)
    
    # Select random indices to display images
    indices = np.random.RandomState(seed=n_seed).choice(len(dataframe), num_images, replace=False)
    
    for i, index in enumerate(indices):
        # Get image data from the column
        image_data = dataframe[image_column][index]
        img_exts = ['jpg', 'jpeg', 'png', 'tiff', 'bmp', 'gif', 'heif', 'raw', 'webp', 'svg', 'psd', 'ico', 'pdf']

        # Check if image data is a file path or a string of image data
        if isinstance(image_data, str) and any(image_data.endswith(ext) for ext in img_exts):
            # If it's a file path
            image = cv2.imread(parent_path + image_data, cv2.IMREAD_GRAYSCALE)
        elif isinstance(image_data, str):
            # If it's a string of image data
            image = np.fromstring(image_data, sep=' ').astype(np.float32)
            image = image.reshape(image_size[0], image_size[1])
        elif len(image_data.shape) == 1:
            # If image data is an array of numbers
            image = np.array(image_data).astype(np.float32)
            image = image.reshape(image_size[0], image_size[1])
        else:
            image = np.array(image_data).astype(np.float32)
        
        # Display image in subplot
        plt.subplot(num_rows, num_columns, i + 1)
        plt.imshow(image, cmap='gray')
        
        # Create dictionary of keypoints
        keypoints = {
            key: (dataframe.loc[index, key], dataframe.loc[index, key.replace('x', 'y')])
            for key in keypoints_columns if re.search(r'x$', key, flags=re.I)
        }

        # Add keypoints to the image
        for key, (x, y) in keypoints.items():
            plt.scatter(x, y, s=scatter_size, marker='.', c=scatter_color)
        
        # Hide axis
        plt.axis('off')
    
    # Show figure
    plt.tight_layout()
    plt.show()