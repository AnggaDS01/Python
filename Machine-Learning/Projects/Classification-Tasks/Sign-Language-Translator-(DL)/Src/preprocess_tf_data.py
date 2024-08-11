import tensorflow as tf
import os

class PreprocessingPipeline:
    def __init__(self, target_size=(224, 224), is_gray=True):
        """
        Initializes the preprocessing pipeline with target image size and color mode.

        Args:
            target_size (tuple): Target size (height, width) for resizing images. Default is (224, 224).
            is_gray (bool): Whether to process images as grayscale or RGB. Default is True (grayscale).
        """
        self.target_size = target_size
        self.is_gray = is_gray

    def __convert_image(self, image_path, label_idx_from_path=None):
        """
        Private method to convert image path to a tensor image.
        Optionally extracts a label from the path if label_idx_from_path is provided.

        Args:
            image_path (str): Path to the image file.
            label_idx_from_path (int, optional): Index to extract the label from the path. Defaults to None.

        Returns:
            image (tf.Tensor): The processed image tensor.
            label (tf.Tensor, optional): The label tensor if label_idx_from_path is provided.
        """
        channels = 1 if self.is_gray else 3
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=channels)
        image.set_shape([None, None, channels])
        image = tf.image.resize(image, size=(self.target_size[0], self.target_size[1]))
        image = image / 255.0

        if label_idx_from_path is not None:
            split_img_path = tf.strings.split(image_path, os.path.sep)
            label = split_img_path[label_idx_from_path]
            return image, label
        else:
            return image

    @tf.autograph.experimental.do_not_convert
    def convert_path_to_img_tf_data_train(self, image_path, label_idx_from_path):
        """
        Converts image path to a tensor image and label for training data.

        Args:
            image_path (tf.Tensor): Tensor containing image path.
            label_idx_from_path (int): Index of the label in the path.

        Returns:
            image (tf.Tensor): The processed image tensor.
            label (tf.Tensor): The label tensor.
        """
        return self.__convert_image(image_path, label_idx_from_path)

    @tf.autograph.experimental.do_not_convert
    def convert_path_to_img_tf_data_test(self, image_path):
        """
        Converts image path to a tensor image for test data without label.

        Args:
            image_path (tf.Tensor): Tensor containing image path.

        Returns:
            image (tf.Tensor): The processed image tensor.
        """
        return self.__convert_image(image_path)

    @tf.autograph.experimental.do_not_convert
    def one_hot_encode(self, image, label, num_classes):
        """
        Converts the label to one-hot encoding.

        Args:
            image (tf.Tensor): The image tensor.
            label (tf.Tensor): The label tensor.
            num_classes (int): Number of classes for one-hot encoding.

        Returns:
            image (tf.Tensor): The image tensor.
            label (tf.Tensor): The one-hot encoded label tensor.
        """
        label = tf.argmax(tf.equal(label, num_classes))
        label = tf.one_hot(label, depth=len(num_classes), dtype=tf.float32)
        return image, label

    @tf.autograph.experimental.do_not_convert
    def augment_image(self, image, label):
        """
        Applies random augmentations to the image.

        Args:
            image (tf.Tensor): The image tensor.
            label (tf.Tensor): The label tensor.

        Returns:
            image (tf.Tensor): The augmented image tensor.
            label (tf.Tensor): The label tensor.
        """
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.clip_by_value(image, clip_value_min=0., clip_value_max=1.)
        return image, label

    @tf.autograph.experimental.do_not_convert
    def get_nan_in_data(self, image, label):
        """
        Checks if there are any NaN values in the label.

        Args:
            image (tf.Tensor): The image tensor.
            label (tf.Tensor): The label tensor.

        Returns:
            has_nan (tf.Tensor): Boolean tensor indicating if NaN values are present.
        """
        return tf.reduce_any(tf.math.is_nan(label))

    @tf.autograph.experimental.do_not_convert
    def processing_data_train_to_VGG_input(self, image, label):
        """
        Converts training data from grayscale to RGB for VGG input.

        Args:
            image (tf.Tensor): The grayscale image tensor.
            label (tf.Tensor): The label tensor.

        Returns:
            image_rgb (tf.Tensor): The RGB image tensor.
            label (tf.Tensor): The label tensor.
        """
        image_rgb = tf.image.grayscale_to_rgb(image)
        return image_rgb, label

    @tf.autograph.experimental.do_not_convert
    def processing_data_test_to_VGG_input(self, image):
        """
        Converts test data from RGB to grayscale and back to RGB for VGG input.

        Args:
            image (tf.Tensor): The RGB image tensor.

        Returns:
            image_rgb (tf.Tensor): The RGB image tensor.
        """
        image_to_gray = tf.image.rgb_to_grayscale(image)
        image_rgb = tf.image.grayscale_to_rgb(image_to_gray)
        return image_rgb
