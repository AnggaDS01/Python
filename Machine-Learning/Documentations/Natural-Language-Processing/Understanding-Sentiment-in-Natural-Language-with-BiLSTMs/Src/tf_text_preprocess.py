import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TokenizerModule:
    def __init__(self, vocab_size=10000, oov_token="<OOV>", max_pad=100):
        """
        Initialize the TokenizerModule with the given parameters.

        Args:
            vocab_size (int): Maximum number of words in the vocabulary.
            oov_token (str): Token for out-of-vocabulary words.
            max_pad (int): Maximum length of padded sequences.
        """
        self.vocab_size = vocab_size
        self.oov_token = oov_token
        self.max_pad = max_pad
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=self.oov_token)

    def fit(self, texts):
        """
        Fit the tokenizer on the provided texts.

        Args:
            texts (list): List of texts to fit the tokenizer on.
        """
        self.tokenizer.fit_on_texts(texts)

    def tokenize_and_pad(self, text):
        """
        Tokenize and pad a single text string.

        Args:
            text (tf.Tensor): Input text to be tokenized and padded.

        Returns:
            tf.Tensor: Tokenized and padded text as a tensor.
        """
        text_lower = tf.strings.lower(text)
        byte_to_text = tf.compat.as_text(text_lower.numpy())
        sequences = self.tokenizer.texts_to_sequences([byte_to_text])
        padded = pad_sequences(sequences, maxlen=self.max_pad, padding='post')
        return tf.convert_to_tensor(padded[0], dtype=tf.int32)

    def tf_tokenize_and_pad(self, text, label):
        """
        Wrapper function to use with tf.data.Dataset for tokenizing and padding.

        Args:
            text (tf.Tensor): Input text to be tokenized and padded.
            label (tf.Tensor): Corresponding label.

        Returns:
            tuple: Tokenized and padded text, and label as a tuple of tensors.
        """
        label = tf.cast(label, tf.int32)
        text_tokenized = tf.py_function(self.tokenize_and_pad, inp=[text], Tout=(tf.int32))
        text_tokenized.set_shape((self.max_pad,))
        return text_tokenized, label