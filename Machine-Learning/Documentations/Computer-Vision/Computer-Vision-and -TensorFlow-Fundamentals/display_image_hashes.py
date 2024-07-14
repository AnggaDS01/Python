from PIL import Image
import matplotlib.pyplot as plt
import imagehash
import numpy as np
from scipy.spatial import distance
import os

def visualize_image_hashes(path_image1, path_image2, figsize:tuple=(10,5)):
    """
    Visualize two images and their perceptual hashes, and calculate the Hamming distance between the hashes.

    Args:
        path_image1 (str): File path to the first image.
        path_image2 (str): File path to the second image.
        figsize (tuple, optional): Size of the figure for displaying images and hashes. Default is (10, 5).

    Returns:
        None

    Example:
        image_path1 = './assets/car1.png'
        image_path2 = './assets/car2.png'
        visualize_image_hashes(image_path1, image_path2)
    """
    # Load images and compute their hashes
    image1 = Image.open(path_image1)
    image2 = Image.open(path_image2)
    images = (image1, image2)
    hashes = [imagehash.phash(img) for img in images]
    
    # Calculate the Hamming distance
    hamming_distance = distance.hamming(hashes[0].hash.flatten(), hashes[1].hash.flatten())
    
    # Visualize images and their hashes
    fig, axes = plt.subplots(2, len(images), figsize=figsize)
    
    for i, (img, hash_val) in enumerate(zip(images, hashes)):
        # Display images
        axes[0, i].imshow(img)
        axes[0, i].set_title(os.path.basename([path_image1, path_image2][i]))
        axes[0, i].axis('off')
        
        # Display hashes
        axes[1, i].imshow(hash_val.hash, cmap='gray')
        axes[1, i].set_title(f"Hash: {os.path.basename([path_image1, path_image2][i])}")
    
    plt.tight_layout()
    plt.show()

    diff_hash = hashes[0].hash != hashes[1].hash
    plt.suptitle(f'Hamming Distance: {hamming_distance * hashes[0].hash.size}')
    plt.imshow(diff_hash, cmap='gray')
    plt.title('Differences')
    plt.axis('off')
    plt.show()

# Contoh penggunaan fungsi
image_path1 = './assets/car1.png'
image_path2 =  './assets/car2.png'
visualize_image_hashes(image_path1, image_path2)
