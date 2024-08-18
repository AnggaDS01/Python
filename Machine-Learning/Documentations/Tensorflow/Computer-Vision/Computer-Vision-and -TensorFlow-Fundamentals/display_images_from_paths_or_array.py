import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math
import os

def display_images(images, titles=None, figsize_per_image:tuple=(3, 3), is_gray:bool=False, is_axis=False, show_histogram=False):
    """
    Display multiple images in a grid layout using PIL and Matplotlib.

    Args:
        images (list): List of image file paths or image arrays to be displayed.
        titles (list, optional): List of titles for each image. Default is None.
        figsize_per_image (tuple, optional): Size of each image in the grid (width, height). Default is (3, 3).
        is_paths (bool, optional): Whether the input is a list of image paths. If False, expects list of image arrays. Default is False.
        is_gray (bool, optional): Whether to convert images to grayscale. Only used if is_paths is True. Default is False.
        is_axis (bool, optional): Whether to display axis on the images. Default is False.
        show_histogram (bool, optional): Whether to display histograms alongside images. Default is False.

    Returns:
        None

    Example:
        image_paths = ['image1.jpg', 'image2.png', 'image3.bmp']
        display_images(image_paths, is_paths=True, titles=['Image 1', 'Image 2', 'Image 3'])

        image_arrays = [translated_img, rotated_img, rescaled_img, sheared_img]
        display_images(image_arrays, titles=['Translated', 'Rotated', 'Rescaled', 'Sheared'])
    """
    
    def load_image(path):
        img = Image.open(path)
        return img.convert('L') if is_gray else img
    
    # Check if input images are paths or arrays
    is_paths = isinstance(images[0], str)

    if is_paths:
        images = [load_image(path) for path in images]
    
    num_images = len(images)
    
    # Calculate the number of rows and columns needed
    num_columns = math.ceil(math.sqrt(num_images))
    num_rows = math.ceil(num_images / num_columns)
    figsize = (num_columns * (figsize_per_image[0] * (2 if show_histogram else 1)), num_rows * figsize_per_image[1])
    
    # Create a figure
    plt.figure(figsize=figsize)
    
    for i, img in enumerate(images):
        plt.subplot(num_rows, num_columns * (2 if show_histogram else 1), (2 * i + 1 if show_histogram else i + 1))
        if isinstance(img, np.ndarray):
            plt.imshow(img, cmap='gray' if is_gray else None)
        else:
            plt.imshow(img, cmap='gray' if is_gray else None)
        plt.title(titles[i] if titles else None)
        plt.axis('on' if is_axis else 'off')
        
        if show_histogram:
            plt.subplot(num_rows, num_columns * 2, 2 * i + 2)
            if isinstance(img, Image.Image):
                img = np.array(img)
            if img.ndim == 2:
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                plt.plot(hist, color='black')
            else:
                color = ('b', 'g', 'r')
                for j, col in enumerate(color):
                    hist = cv2.calcHist([img], [j], None, [256], [0, 256])
                    plt.plot(hist, color=col)
            plt.grid()
            plt.xlim([0, 256])
            plt.xlabel('color intensity value (0 to 256)')
            plt.ylabel('frequency of occurrence')
            plt.title('Histogram')

    plt.tight_layout()
    plt.show()