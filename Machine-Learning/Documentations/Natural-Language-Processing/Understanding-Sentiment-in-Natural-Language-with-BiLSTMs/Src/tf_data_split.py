import tensorflow as tf

class DatasetSplitter:
    def __init__(self, batch_size=64, train_split=0.9, shuffle_buffer_size=None):
        """
        Initialize the DatasetSplitter with default or provided parameters.

        Args:
            batch_size (int, optional): The size of the batches. Default is 64.
            train_split (float, optional): The proportion of the dataset to use for training. Default is 0.9.
            shuffle_buffer_size (int, optional): The buffer size for shuffling the dataset. Default is None.
        """
        self.batch_size = batch_size
        self.train_split = train_split
        self.shuffle_buffer_size = shuffle_buffer_size

    def split_and_prepare(self, dataset):
        """
        Split a dataset into training and validation sets, batch, and prefetch them.

        Args:
            dataset (tf.data.Dataset): The dataset to be split and prepared.

        Returns:
            tuple: A tuple containing the training and validation datasets, both batched and prefetched.
        """
        dataset_shuffled = self._shuffle_dataset(dataset)
        dataset_batched_and_prefetched = self._batch_and_prefetch(dataset_shuffled)

        train_size = int(self.train_split * len(dataset_batched_and_prefetched))
        train_dataset = dataset_batched_and_prefetched.take(train_size)
        valid_dataset = dataset_batched_and_prefetched.skip(train_size)

        self._display_info(dataset, train_dataset, valid_dataset)

        return train_dataset, valid_dataset

    def _shuffle_dataset(self, dataset):
        if self.shuffle_buffer_size is None:
            self.shuffle_buffer_size = len(dataset)
        return dataset.shuffle(self.shuffle_buffer_size)

    def _batch_and_prefetch(self, dataset):
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def _display_info(self, dataset, train_dataset, valid_dataset):
        print(f"=================================== Original Dataset ===================================")
        print(f"Info data: {dataset}")
        print(f"Number of data: {len(dataset)}")
        print(f"BATCH SIZE: {self.batch_size}")

        print(f"=================================== Training Dataset ===================================")
        print(f"Info data: {train_dataset}")
        print(f"Training Split: {self.train_split}")
        print(f"Number of data: {len(train_dataset)}")

        print(f"=================================== Validation Dataset ===================================")
        print(f"Info data: {valid_dataset}")
        print(f"Validation Split: {round(1 - self.train_split, 2)}")
        print(f"Number of data: {len(valid_dataset)}")