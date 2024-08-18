import numpy as np
import cv2

def add_noise(image, noise_type="gaussian", noise_level=25):
    """
    Add random noise to an image.
    
    Args:
        image (ndarray): Input image.
        noise_type (str): Type of noise to add ('gaussian' or 'salt_pepper').
        noise_level (int): Noise level. Higher values mean more noise.
    
    Returns:
        noisy_image (ndarray): Noisy image.
    """
    if noise_type == "gaussian":
        mean = 0
        sigma = noise_level
        gaussian_noise = np.random.normal(mean, sigma, image.shape).astype('uint8')
        noisy_image = cv2.add(image, gaussian_noise)
    elif noise_type == "salt_pepper":
        s_vs_p = 0.5
        amount = noise_level / 100.0
        noisy_image = image.copy()
        
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
        noisy_image[tuple(coords)] = 255  # Assuming image is in 8-bit format

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
        noisy_image[tuple(coords)] = 0
    return noisy_image