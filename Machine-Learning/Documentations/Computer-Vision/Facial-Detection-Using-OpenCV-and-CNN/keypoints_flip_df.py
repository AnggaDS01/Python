import pandas as pd
import numpy as np
import cv2
import re

def flip_coordinate_and_image_horizontal_in_df(df, image_width, image_height, image_column):
    """
    Apply horizontal flip to keypoints and images in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the keypoints and image paths or data.
        image_width (int): The size of the image.
        image_column (str): The column name in the DataFrame containing the image paths or data.

    Returns:
        pd.DataFrame: The DataFrame with horizontally flipped keypoints and images.

    Example:
        >>> import pandas as pd
        >>> from keypoints_flip_df import flip_coordinate_and_image_horizontal_in_df
        >>>
        >>> # Create a mock DataFrame for demonstration
        >>> data = {'image': ['img1.jpg', 'img2.jpg'],
        >>>         'nose_x': [30, 40],
        >>>         'nose_y': [50, 60]}
        >>> df = pd.DataFrame(data)
        >>>
        >>> # Apply horizontal flip
        >>> flipped_df = flip_coordinate_and_image_horizontal(df, 96, 'image')
        >>> print(flipped_df)
    """
    keypoint_columns = [col for col in df.columns if re.search(r'x$', col, flags=re.I)]

    for col in keypoint_columns:
        df[col] = df[col].apply(lambda x: image_width - 1 - x)

    if image_column in df.columns:
        for i, row in df.iterrows():
            image_path_or_data = row[image_column]
            img_exts = ['jpg', 'jpeg', 'png', 'tiff', 'bmp', 'gif', 'heif', 'raw', 'webp', 'svg', 'psd', 'ico', 'pdf']

            # Load image
            if isinstance(image_path_or_data, str) and any(image_path_or_data.endswith(ext) for ext in img_exts):
                # If path to the image file
                image = cv2.imread(image_path_or_data, cv2.IMREAD_GRAYSCALE)
            else:
                # If image data in string format
                image = np.fromstring(image_path_or_data, sep=' ').astype(np.float32).reshape(image_height, image_width)
            
            # Flip the image
            flipped_image = cv2.flip(image, 1)

            df.at[i, image_column] = flipped_image
            
    return df

def flip_coordinate_and_image_vertical_in_df(df, image_width, image_height, image_column):
    """
    Apply vertical flip to keypoints and images in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the keypoints and image paths or data.
        image_size (int): The size of the image.
        image_column (str): The column name in the DataFrame containing the image paths or data.

    Returns:
        pd.DataFrame: The DataFrame with vertically flipped keypoints and images.

    Example:
        >>> import pandas as pd
        >>> from keypoints_flip_df import flip_coordinate_and_image_vertical_in_df
        >>>
        >>> # Create a mock DataFrame for demonstration
        >>> data = {'image': ['img1.jpg', 'img2.jpg'],
        >>>         'nose_x': [30, 40],
        >>>         'nose_y': [50, 60]}
        >>> df = pd.DataFrame(data)
        >>>
        >>> # Apply vertical flip
        >>> flipped_df = flip_coordinate_and_image_vertical(df, 96, 'image')
        >>> print(flipped_df)
    """
    keypoint_columns = [col for col in df.columns if re.search(r'y$', col, flags=re.I)]

    for col in keypoint_columns:
        df[col] = df[col].apply(lambda y: image_width - 1 - y)

    if image_column in df.columns:
        for i, row in df.iterrows():
            image_path_or_data = row[image_column]
            img_exts = ['jpg', 'jpeg', 'png', 'tiff', 'bmp', 'gif', 'heif', 'raw', 'webp', 'svg', 'psd', 'ico', 'pdf']

            # Load image
            if isinstance(image_path_or_data, str) and any(image_path_or_data.endswith(ext) for ext in img_exts):
                # If path to the image file
                image = cv2.imread(image_path_or_data, cv2.IMREAD_GRAYSCALE)
            else:
                # If image data in string format
                image = np.fromstring(image_path_or_data, sep=' ').astype(np.float32).reshape(image_height, image_width)
            
            # Flip the image
            flipped_image = cv2.flip(image, 0)

            df.at[i, image_column] = flipped_image
            
    return df