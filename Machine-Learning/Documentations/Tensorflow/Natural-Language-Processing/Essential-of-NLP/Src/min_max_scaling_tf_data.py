import tensorflow as tf

def minmax_scaling_tf(data, label, feature_range=(0, 1)):
    data = tf.cast(data, tf.float32)
    label = tf.cast(label, tf.uint8)
    
    min_val = tf.reduce_min(data, axis=0)
    max_val = tf.reduce_max(data, axis=0)
    scale = (feature_range[1] - feature_range[0]) / (max_val - min_val)
    scaled_data = feature_range[0] + scale * (data - min_val)
    return scaled_data, label