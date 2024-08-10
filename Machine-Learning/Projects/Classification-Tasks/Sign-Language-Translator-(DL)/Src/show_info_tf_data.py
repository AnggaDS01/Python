import tensorflow as tf
import os

class FilePathInfo:
    def __init__(self, unit_file_size='bytes'):
        """
        Initialize the FilePathInfo class with default or provided parameters.

        Args:
            unit_file_size (str, optional): The unit for displaying file sizes ('bytes', 'kb', 'mb', 'gb'). Default is 'bytes'.
        """
        self.unit_file_size = unit_file_size.lower()
        self.units = ['bytes', 'kb', 'mb', 'gb']
        if self.unit_file_size not in self.units:
            raise ValueError(f"Invalid unit. Choose from {self.units}.")

    def show_train_files_path_info(self, files_path_data, kind_data, is_random=False):
        """
        Display detailed information about the file paths in the training dataset.

        Args:
            files_path_data (tf.data.Dataset): The dataset containing file paths.
            kind_data (str): The specific kind of data to locate within the file paths.
            is_random (bool, optional): Whether to randomly shuffle the dataset before displaying. Default is False.
        """
        files_path_data_plot = self._get_files_path_data_plot(files_path_data, is_random)
        self._display_path_info(files_path_data_plot, kind_data)

    def show_test_files_path_info(self, files_path_data, is_random=False):
        """
        Display detailed information about the file paths in the testing dataset.

        Args:
            files_path_data (tf.data.Dataset): The dataset containing file paths.
            is_random (bool, optional): Whether to randomly shuffle the dataset before displaying. Default is False.
        """
        files_path_data_plot = self._get_files_path_data_plot(files_path_data, is_random)
        self._display_path_info(files_path_data_plot)

    def _get_files_path_data_plot(self, files_path_data, is_random):
        if is_random:
            return files_path_data.take(1)
        else:
            return files_path_data.shuffle(buffer_size=files_path_data.cardinality().numpy()).take(1)

    def _display_path_info(self, files_path_data_plot, kind_data=None):
        for file_path in files_path_data_plot:
            print('=' * 60)
            print(' PATH INFO '.center(60, '='))
            print('=' * 60)
            print(f'File Path: {file_path.numpy().decode("utf-8")}')
            print()

            split_file_path = self._split_file_path(file_path)
            self._display_split_file_path(split_file_path)

            if kind_data:
                self._display_kind_data_info(split_file_path, kind_data)

            self._display_file_info(split_file_path, file_path)

    def _split_file_path(self, file_path):
        return tf.strings.split(file_path, os.path.sep)

    def _display_split_file_path(self, split_file_path):
        print('=' * 60)
        print(' SPLIT FILE PATH '.center(60, '='))
        print('=' * 60)
        print(f'Split File Path: {split_file_path}')
        print()

        print('=' * 60)
        print(' INDEXED PATH '.center(60, '='))
        print('=' * 60)
        result = {value: f'Index -> {index}' for index, value in enumerate(split_file_path.numpy())}
        for key, value in result.items():
            print(f'{value}: {key}')
        print()

    def _display_kind_data_info(self, split_file_path, kind_data):
        print('=' * 60)
        print(f' KIND DATA INDEX: {kind_data} '.center(60, '='))
        print('=' * 60)
        index = tf.where(tf.equal(split_file_path, kind_data))[0][0]
        print(f'Index of "{kind_data}": {index}')
        print()

        index_label = index + 1
        print('=' * 60)
        print(' INDEX LABEL '.center(60, '='))
        print('=' * 60)
        print(f'Index Label: {index_label}')
        print()

        print('=' * 60)
        print(' LABEL '.center(60, '='))
        print('=' * 60)
        print(f'Label: {split_file_path[index_label]}')
        print()

    def _display_file_info(self, split_file_path, file_path):
        file_name = split_file_path[-1].numpy().decode('utf-8')
        print('=' * 60)
        print(' FILE NAME '.center(60, '='))
        print('=' * 60)
        print(f'File Name: {file_name}')
        print()

        file_extension = os.path.splitext(file_name)[1]
        print('=' * 60)
        print(' FILE EXTENSION '.center(60, '='))
        print('=' * 60)
        print(f'File Extension: {file_extension}')
        print()

        file_size = os.path.getsize(file_path.numpy().decode('utf-8'))
        file_size = self._format_file_size(file_size)
        print('=' * 60)
        print(' FILE SIZE '.center(60, '='))
        print('=' * 60)
        print(f'File Size: {file_size} {self.unit_file_size}')
        print()

    def _format_file_size(self, size):
        if self.unit_file_size == 'kb':
            size /= 1024
        elif self.unit_file_size == 'mb':
            size /= 1024 ** 2
        elif self.unit_file_size == 'gb':
            size /= 1024 ** 3

        return f'{size:.4f}' if self.unit_file_size != 'bytes' else size
