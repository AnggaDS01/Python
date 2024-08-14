import tensorflow as tf

class DatasetSplitter:
    def __init__(self, batch_size=64, train_split=0.9, shuffle_buffer_size=None, seed=None):
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
        self.seed = seed

    def split_and_prepare(self, dataset):
        """
        Split a dataset into training and validation sets, batch, and prefetch them.

        Args:
            dataset (tf.data.Dataset): The dataset to be split and prepared.

        Returns:
            tuple: A tuple containing the training and validation datasets, both batched and prefetched.
        """
        tf.random.set_seed(self.seed)

        dataset_shuffled = self._shuffle_dataset(dataset)
        train_size = int(self.train_split * len(dataset))

        train_dataset = dataset_shuffled.take(train_size)
        valid_dataset = dataset_shuffled.skip(train_size)

        train_dataset_batched = self._batch_and_prefetch(train_dataset)
        valid_dataset_batched = self._batch_and_prefetch(valid_dataset)

        self._display_info(train_dataset, train_dataset_batched, valid_dataset, valid_dataset_batched)

        return train_dataset_batched, valid_dataset_batched

    def _shuffle_dataset(self, dataset):
        if self.shuffle_buffer_size is None:
            self.shuffle_buffer_size = len(dataset)
        return dataset.shuffle(self.shuffle_buffer_size, seed=self.seed)

    def _batch_and_prefetch(self, dataset):
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def _display_info(self, train_dataset, train_dataset_batched, valid_dataset, valid_dataset_batched):
        print(f"=================================== Training Dataset ===================================")
        print(f"Info data: {train_dataset}")
        print(f"Training Split: {self.train_split}")
        print(f"Number of data: {len(train_dataset)}")
        print(f"AFTER BATCH: {self.batch_size}")
        print(f"Number of data: {len(train_dataset_batched)}")

        print(f"=================================== Validation Dataset ===================================")
        print(f"Info data: {valid_dataset}")
        print(f"Validation Split: {round(1 - self.train_split, 2)}")
        print(f"Number of data: {len(valid_dataset)}")
        print(f"AFTER BATCH: {self.batch_size}")
        print(f"Number of data: {len(valid_dataset_batched)}")