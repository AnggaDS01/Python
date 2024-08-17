import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class ReviewAnalysis:
    def __init__(self, tokenizer_module):
        """
        Initialize the ReviewAnalysis class.

        Args:
            tokenizer_module (TokenizerModule): An instance of TokenizerModule class used for tokenization.

        Example:
            tokenizer_module = TokenizerModule(vocab_size=10000, oov_token="<OOV>", max_pad=100)
            review_analysis = ReviewAnalysis(tokenizer_module)
        """
        self.tokenizer_module = tokenizer_module

    def _calculate_length(self, text):
        """
        Private method to calculate the length of a tokenized text.

        Args:
            text (tf.Tensor): The input text tensor.

        Returns:
            int: The length of the tokenized sequence.
        """
        text_lower = tf.strings.lower(text) 
        byte_to_text = tf.compat.as_text(text_lower.numpy())
        sequences = self.tokenizer_module.tokenizer.texts_to_sequences([byte_to_text])
        length = len(sequences[0])
        return length

    def tf_calculate_length(self, text, label):
        """
        Method to calculate the length of a text sequence in a TensorFlow dataset.

        Args:
            text (tf.Tensor): The input text tensor.
            label (tf.Tensor): The label tensor (not used in length calculation).

        Returns:
            tf.Tensor: The length of the tokenized sequence as a tensor.
        """
        length = tf.py_function(self._calculate_length, inp=[text], Tout=tf.int64)
        return length

    def get_review_lengths(self, dataset):
        """
        Method to get the lengths of all reviews in a dataset.

        Args:
            dataset (tf.data.Dataset): The TensorFlow dataset containing text reviews and labels.

        Returns:
            tf.data.Dataset: A dataset containing the lengths of the reviews.
        """
        return dataset.map(self.tf_calculate_length, num_parallel_calls=tf.data.AUTOTUNE).cache()

    def compute_length_statistics(self, review_lengths):
        """
        Method to compute length statistics (sum, max, min, average) for review lengths.

        Args:
            review_lengths (tf.data.Dataset): A dataset containing the lengths of reviews.

        Returns:
            tuple: A tuple containing sum, max, min, and average lengths of reviews.
        """
        sum_review_lengths = review_lengths.reduce(
            initial_state=tf.constant(0, dtype=tf.int64),
            reduce_func=lambda x, y: x + y
        )

        max_token_length = review_lengths.reduce(
            initial_state=tf.constant(0, dtype=tf.int64),
            reduce_func=lambda x, y: tf.maximum(x, y)
        )

        min_token_length = review_lengths.reduce(
            initial_state=tf.constant(9**10, dtype=tf.int64),
            reduce_func=lambda x, y: tf.minimum(x, y)
        )

        average_token_length = sum_review_lengths.numpy() / len(review_lengths)
        
        return max_token_length, min_token_length, average_token_length

    def _length_filter_review(self, text, length_filter, operator):
        """
        Private method to filter reviews based on their length.

        Args:
            text (tf.Tensor): The input text tensor.
            length_filter (int): The length threshold for filtering.
            operator (tf.Tensor): The comparison operator as a string.

        Returns:
            bool: True if the condition is met, False otherwise.
        """
        length = self._calculate_length(text)
        condition = eval(f"{length} {tf.compat.as_text(operator.numpy())} {length_filter}")
        return condition

    def tf_length_review(self, text, label, length_filter, operator):
        """
        Method to filter reviews in a TensorFlow dataset based on their length.

        Args:
            text (tf.Tensor): The input text tensor.
            label (tf.Tensor): The label tensor.
            length_filter (int): The length threshold for filtering.
            operator (str): The comparison operator as a string (e.g., "==", ">=", "<=").

        Returns:
            tf.Tensor: A boolean tensor indicating whether the review passes the filter.
        """
        result = tf.py_function(self._length_filter_review, inp=[text, length_filter, operator], Tout=tf.bool)
        result.set_shape([])
        return result

    def filter_reviews(self, dataset, length_filter, operator=">"):
        """
        Method to filter a TensorFlow dataset based on review length.

        Args:
            dataset (tf.data.Dataset): The TensorFlow dataset containing text reviews and labels.
            length_filter (int): The length threshold for filtering.
            operator (str): The comparison operator as a string (e.g., "==", ">", "<").

        Returns:
            tf.data.Dataset: A filtered dataset containing only reviews that meet the length condition.

        Example:
            filtered_dataset = review_analysis.filter_reviews(imdb_reviews_tf_dataset, 6, ">")
        """
        return dataset.filter(lambda text, label: self.tf_length_review(text, label, length_filter, operator)).cache()