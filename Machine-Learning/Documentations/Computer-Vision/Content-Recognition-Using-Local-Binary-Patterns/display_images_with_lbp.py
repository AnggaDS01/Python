import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import local_binary_pattern

def display_images_with_lbp(images, radius=3, eps=1e-7, titles=None, figsize_per_image=(3, 3), is_axis=False, get_histogram=False):
    """
    Display original images, LBP images, and LBP histograms in a grid layout using PIL and Matplotlib.
    
    Args:
        images (list): List of image file paths or image arrays to be displayed.
        radius (int, optional): Radius parameter for LBP calculation. Default is 3.
        eps (float, optional): Small value to avoid division by zero in histogram normalization. Default is 1e-7.
        titles (list, optional): List of titles for each image. Default is None.
        figsize_per_image (tuple, optional): Size of each image in the grid (width, height). Default is (3, 3).
        is_axis (bool, optional): Whether to display axis on the images. Default is False.
        get_histogram (bool, optional): Whether to return histograms without plotting them. Default is False.
    
    Returns:
        histograms (list): List of histograms if get_histogram is True, otherwise None.
    """
    
    def load_image(path):
        """Load an image from a file path and convert it to a NumPy array."""
        img = Image.open(path)
        return np.array(img)
    
    def calculate_lbp_histogram(image):
        """Calculate the LBP and its histogram for a given image."""
        n_points = int((8 * radius + 16) / 3)  # Calculate the number of points for LBP
        lbp = local_binary_pattern(image, n_points, radius, method="uniform")  # Compute LBP
        # Compute the histogram of the LBP result
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")  # Convert histogram to float
        hist /= (hist.sum() + eps)  # Normalize the histogram
        return lbp, hist
    
    # Check if input images are paths or arrays
    is_paths = isinstance(images[0], str)
    
    if is_paths:
        images = [load_image(path) for path in images]  # Load images if paths are provided
    
    num_images = len(images)
    num_columns = 3  # For original, LBP, and histogram
    num_rows = num_images
    figsize = (num_columns * figsize_per_image[0], num_rows * figsize_per_image[1])  # Calculate figure size
    
    histograms = [] if get_histogram else None
    
    if not get_histogram: 
        plt.figure(figsize=figsize)  # Create a figure for plotting
    
    for i, img in enumerate(images):
        # Convert image to grayscale if it is not already
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp, hist = calculate_lbp_histogram(img)  # Calculate LBP and histogram
        
        if not get_histogram:
            # Plot original image
            plt.subplot(num_rows, num_columns, i * num_columns + 1)
            plt.imshow(img, cmap='gray')
            plt.title(titles[i] if titles else "Original")
            plt.axis('on' if is_axis else 'off')
            
            # Plot LBP image
            plt.subplot(num_rows, num_columns, i * num_columns + 2)
            plt.imshow(lbp, cmap='gray')
            plt.title("LBP")
            plt.axis('on' if is_axis else 'off')
        
            # Plot histogram
            plt.subplot(num_rows, num_columns, i * num_columns + 3)
            n_bins = int(lbp.max() + 1)
            plt.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins), edgecolor='black', linewidth=.5)
            plt.title("Histogram")
            plt.xlabel('Uniform LBP value')
            plt.ylabel('Percentage')
            plt.grid(True)
        else:
            histograms.append(hist)  # Store histogram if get_histogram is True
    
    if not get_histogram:
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()  # Show the plot
    
    return histograms  # Return histograms if get_histogram is True