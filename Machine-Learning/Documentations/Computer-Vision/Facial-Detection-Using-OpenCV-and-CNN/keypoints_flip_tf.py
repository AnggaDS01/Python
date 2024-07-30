import tensorflow as tf

def flip_coordinate_and_image_horizontal_in_tf_dataset(image, keypoint):
    """
    Flip the image and keypoints horizontally.

    Args:
        image (tf.Tensor): Tensor representing the image to be flipped.
        keypoint (tf.Tensor): Tensor containing the keypoints corresponding to the image.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Flipped image and keypoints.

    Example:
        >>> import tensorflow as tf
        >>> from keypoints_flip_tf import flip_coordinate_and_image_horizontal_in_tf_dataset
        >>>
        >>> # Create a mock image and keypoints for demonstration
        >>> image = tf.random.uniform((96, 96, 1), minval=0, maxval=1)
        >>> keypoints = tf.constant([10, 20, 30, 40], dtype=tf.float32)
        >>>
        >>> flipped_image, flipped_keypoints = flip_coordinate_and_image_horizontal_in_tf_dataset(image, keypoints)
    """

    flipped_keypoints = []
    for i in range(len(keypoint)):
        if i % 2 == 0:
            flipped_keypoints.append(image.shape[1] - 1 - keypoint[i])
        else:
            flipped_keypoints.append(keypoint[i])
    flipped_keypoints = tf.stack(flipped_keypoints)

    image = tf.image.flip_left_right(image)
    return image, flipped_keypoints

def flip_coordinate_and_image_vertical_in_tf_dataset(image, keypoint):
    """
    Flip the image and keypoints vertically.

    Args:
        image (tf.Tensor): Tensor representing the image to be flipped.
        keypoint (tf.Tensor): Tensor containing the keypoints corresponding to the image.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Flipped image and keypoints.

    Example:
        >>> import tensorflow as tf
        >>> from keypoints_flip_tf import flip_coordinate_and_image_vertical_in_tf_dataset
        >>>
        >>> # Create a mock image and keypoints for demonstration
        >>> image = tf.random.uniform((96, 96, 1), minval=0, maxval=1)
        >>> keypoints = tf.constant([10, 20, 30, 40], dtype=tf.float32)
        >>>
        >>> flipped_image, flipped_keypoints = flip_coordinate_and_image_vertical_in_tf_dataset(image, keypoints)
    """

    flipped_keypoints = []
    for i in range(len(keypoint)):
        if i % 2 == 1:
            flipped_keypoints.append(image.shape[0] - 1 - keypoint[i])
        else:
            flipped_keypoints.append(keypoint[i])
    flipped_keypoints = tf.stack(flipped_keypoints)

    image = tf.image.flip_up_down(image)
    return image, flipped_keypoints