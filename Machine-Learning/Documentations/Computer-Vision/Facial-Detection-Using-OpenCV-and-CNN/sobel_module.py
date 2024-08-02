import tensorflow as tf
import numpy as np
import cv2

def compute_sobel(image, keypoint, ksize):
    """
    Compute the Sobel gradient of an image and return it along with the keypoints.

    Args:
        image (tf.Tensor): The input image tensor.
        keypoint (tf.Tensor): The keypoints tensor.
        ksize (int): The size of the extended Sobel kernel. It must be 1, 3, 5, or 7.

    Returns:
        np.ndarray: The Sobel gradient image.
        tf.Tensor: The keypoints tensor.

    """
    # Convert Tensor to NumPy array
    image_np = image.numpy()
    ksize = int(ksize)  # Ensure ksize is an integer

    # Compute Sobel gradients
    sobelX = cv2.Sobel(image_np, cv2.CV_32F, 1, 0, ksize=ksize)
    sobelY = cv2.Sobel(image_np, cv2.CV_32F, 0, 1, ksize=ksize)

    # Compute gradient magnitude
    sobelG = np.hypot(sobelX, sobelY)
    sobelG = np.expand_dims(sobelG, axis=-1)
    return sobelG, keypoint

def tf_compute_sobel(image, keypoint, ksize):
    """
    Compute the Sobel gradient of an image in a TensorFlow graph.

    Args:
        image (tf.Tensor): The input image tensor.
        keypoint (tf.Tensor): The keypoints tensor.
        ksize (int): The size of the extended Sobel kernel. It must be 1, 3, 5, or 7.

    Returns:
        tf.Tensor: The Sobel gradient image tensor.
        tf.Tensor: The keypoints tensor.

    Example:
        >>> import tensorflow as tf
        >>> from sobel_module import tf_compute_sobel
        >>> image = tf.random.normal([96, 96, 1])
        >>> keypoints = tf.random.normal([30])
        >>> sobel_image, keypoints = tf_compute_sobel(image, keypoints, ksize=3)
    """
    sobelG, keypoint = tf.py_function(func=compute_sobel, inp=[image, keypoint, ksize], Tout=[tf.float32, keypoint.dtype])
    sobelG = (sobelG / tf.reduce_max(sobelG)) * 1.0
    return sobelG, keypoint