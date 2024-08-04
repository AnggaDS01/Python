import tensorflow as tf
def split_and_prepare_datasets(dataset, batch_size=64, train_split=0.9, shuffle_buffer_size=None):
    """
    Split a dataset into training and validation sets, batch, and prefetch them.

    Args:
        dataset (tf.data.Dataset): The dataset to be split and prepared.
        batch_size (int, optional): The size of the batches. Default is 64.
        train_split (float, optional): The proportion of the dataset to use for training. Default is 0.9.
        shuffle_buffer_size (int, optional): The buffer size for shuffling the dataset. Default is None.

    Returns:
        dict: A dictionary containing the training and validation datasets, both batched and prefetched.

    Example:
        >>> import tensorflow as tf
        >>> from data_split import split_and_prepare_datasets
        >>>
        >>> # Create a mock dataset for demonstration
        >>> dataset = tf.data.Dataset.from_tensor_slices((tf.random.normal([100, 96, 96, 1]), tf.random.normal([100, 30])))
        >>>
        >>> # Split and prepare the dataset
        >>> datasets_info = split_and_prepare_datasets(dataset, batch_size=32, train_split=0.8, shuffle_buffer_size=100)
        >>> for key, value in datasets_info.items():
        >>>     print(f"{key}: {value}")
    """
    # Shuffle dataset if buffer size is provided
    if shuffle_buffer_size is None:
        shuffle_buffer_size = len(dataset)
    dataset_shuffled = dataset.shuffle(shuffle_buffer_size)
    
    # Determine the size of the training set
    train_size = int(train_split * len(dataset))
    
    # Split the dataset
    train_dataset = dataset_shuffled.take(train_size)
    valid_dataset = dataset_shuffled.skip(train_size)
    
    # Batch and prefetch the datasets
    train_dataset_batched = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    valid_dataset_batched = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Display information about the datasets
    print(f"=================================== Training Dataset ===================================")
    print(f"Info data: {train_dataset}")
    print(f"Training Split: {train_split}")
    print(f"Number of data: {len(train_dataset)}")
    print(f"AFTER BATCH: {batch_size }")
    print(f"Number of data: {len(train_dataset_batched)}")

    print(f"=================================== Validation Dataset ===================================")
    print(f"Info data: {valid_dataset}")
    print(f"Validation Split: {round(1 - train_split, 2)}")
    print(f"Number of data: {len(valid_dataset)}")
    print(f"AFTER BATCH: {batch_size }")
    print(f"Number of data: {len(valid_dataset_batched)}")
    
    return  train_dataset_batched, valid_dataset_batched