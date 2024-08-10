import tensorflow as tf
import os

class PreprocessingPipeline:
    def __init__(self, target_size=(224, 224), is_gray=True):
        self.target_size = target_size
        self.is_gray = is_gray

    def __convert_image(self, image_path, label_idx_from_path=None):
        """
        Private method untuk mengonversi path gambar menjadi tensor image.
        Jika label_idx_from_path diberikan, juga mengambil label dari path.
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
        Public method untuk mengonversi path gambar menjadi tensor image dan label untuk data pelatihan.
        """
        return self.__convert_image(image_path, label_idx_from_path)

    @tf.autograph.experimental.do_not_convert
    def convert_path_to_img_tf_data_test(self, image_path):
        """
        Public method untuk mengonversi path gambar menjadi tensor image untuk data uji tanpa label.
        """
        return self.__convert_image(image_path)

    @tf.autograph.experimental.do_not_convert
    def one_hot_encode(self, image, label, num_classes):
        """
        Public method untuk mengubah label menjadi one-hot encoding.
        """
        label = tf.argmax(tf.equal(label, num_classes))
        label = tf.one_hot(label, depth=len(num_classes), dtype=tf.float32)
        return image, label

    @tf.autograph.experimental.do_not_convert
    def augment_image(self, image, label):
        """
        Public method untuk melakukan augmentasi pada image.
        """
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.clip_by_value(image, clip_value_min=0., clip_value_max=1.)
        return image, label

    @tf.autograph.experimental.do_not_convert
    def get_nan_in_data(self, image, keypoint):
        """
        Public method untuk memeriksa apakah ada NaN dalam data keypoint.
        """
        return tf.reduce_any(tf.math.is_nan(keypoint))

    @tf.autograph.experimental.do_not_convert
    def processing_data_train_to_VGG_input(self, image, label):
        """
        Public method untuk mengonversi data pelatihan dari grayscale ke RGB untuk input VGG.
        """
        image_rgb = tf.image.grayscale_to_rgb(image)
        return image_rgb, label

    @tf.autograph.experimental.do_not_convert
    def processing_data_test_to_VGG_input(self, image):
        """
        Public method untuk mengonversi data uji dari RGB ke grayscale dan kembali ke RGB untuk input VGG.
        """
        image_to_gray = tf.image.rgb_to_grayscale(image)
        image_rgb = tf.image.grayscale_to_rgb(image_to_gray)
        return image_rgb
