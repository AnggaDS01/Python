import tensorflow as tf
import numpy as np
import cv2

def compute_canny(image, keypoint, threshold1, threshold2):
    """
    Compute the Canny edges of an image and return it along with the keypoints.

    Args:
        image (tf.Tensor): The input image tensor.
        keypoint (tf.Tensor): The keypoints tensor.
        threshold1 (int): The first threshold for the hysteresis procedure in Canny edge detection.
        threshold2 (int): The second threshold for the hysteresis procedure in Canny edge detection.

    Returns:
        np.ndarray: The Canny edge image.
        tf.Tensor: The keypoints tensor.

    """
    # Convert Tensor to NumPy array
    image_np = image.numpy()
    threshold1 = int(threshold1)
    threshold2 = int(threshold2)

    # Check if the image is in the range [0, 1]
    if image_np.max() <= 1.0:
        image_np = (image_np * 255)

    # Convert image to uint8
    image_np = cv2.convertScaleAbs(image_np)

    # Compute Canny edges
    canny_edges = cv2.Canny(image_np, threshold1, threshold2)

    # Add extra dimension to match expected shape
    canny_edges = np.expand_dims(canny_edges, axis=-1)

    return canny_edges, keypoint

def tf_compute_canny(image, keypoint, threshold1, threshold2):
    """
    Compute the Canny edges of an image in a TensorFlow graph.

    Args:
        image (tf.Tensor): The input image tensor.
        keypoint (tf.Tensor): The keypoints tensor.
        threshold1 (int): The first threshold for the hysteresis procedure in Canny edge detection.
        threshold2 (int): The second threshold for the hysteresis procedure in Canny edge detection.

    Returns:
        tf.Tensor: The Canny edge image tensor.
        tf.Tensor: The keypoints tensor.

    Example:
        >>> import tensorflow as tf
        >>> from canny_module import tf_compute_canny
        >>> image = tf.random.normal([96, 96, 1])
        >>> keypoints = tf.random.normal([30])
        >>> canny_image, keypoints = tf_compute_canny(image, keypoints, threshold1=100, threshold2=200)
    """
    canny_edges, keypoint_canny = tf.py_function(func=compute_canny, inp=[image, keypoint, threshold1, threshold2], Tout=[tf.float32, keypoint.dtype])
    canny_edges = (canny_edges / 255) * 1.0
    canny_edges.set_shape(image.shape)
    keypoint_canny.set_shape((keypoint.shape))
    return canny_edges, keypoint_canny