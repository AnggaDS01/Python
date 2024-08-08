import tensorflow as tf
import os

def convert_path_to_img_tf_data_train(image_path, label_idx_from_path, target_size, is_gray=True):
    split_img_path = tf.strings.split(image_path, os.path.sep)
    label = split_img_path[label_idx_from_path]

    channels = 1 if is_gray else 3 
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=channels) 
    image.set_shape([None, None, channels])
    image = tf.image.resize(image, size=(target_size[0], target_size[1]))
    image = image / 255.0
    return image, label

def convert_path_to_img_tf_data_test(image_path, target_size, is_gray=True):
    channels = 1 if is_gray else 3 
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=channels) 
    image.set_shape([None, None, channels])
    image = tf.image.resize(image, size=(target_size[0], target_size[1]))
    image = image / 255.0
    return image

def one_hot_encode(image, label, num_classes):
    label = tf.argmax(tf.equal(label, num_classes))
    label = tf.one_hot(label, depth=len(num_classes), dtype=tf.float32)
    return image, label

def augment_image(image, label):
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.clip_by_value(image, clip_value_min=0., clip_value_max=1.)
    return image, label

def get_nan_in_data(image, keypoint):
    get_nan_in_data = tf.reduce_any(tf.math.is_nan(keypoint))
    return get_nan_in_data