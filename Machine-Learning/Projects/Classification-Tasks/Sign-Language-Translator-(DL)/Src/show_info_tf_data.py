import tensorflow as tf
import numpy as np
import os

def format_file_size(size, unit_file_size='bytes'):
    """Format file size to the specified unit."""
    units = ['bytes', 'kb', 'mb', 'gb']
    if unit_file_size.lower() not in units:
        raise ValueError(f"Invalid unit. Choose from {units}.")
    
    if unit_file_size.lower() == 'kb':
        size /= 1024
    elif unit_file_size.lower() == 'mb':
        size /= 1024 ** 2
    elif unit_file_size.lower() == 'gb':
        size /= 1024 ** 3
    
    return f'{size:.4f}' if unit_file_size.lower() != 'bytes' else size

# ==================================================== DATA TRAIN ====================================================
def show_train_files_path_info(files_path_data, kind_data, is_random=False, unit_file_size='bytes'):
    
    if is_random:
        files_path_data_plot = files_path_data.take(1)
    else:
        files_path_data_plot = files_path_data.shuffle(buffer_size=files_path_data.cardinality().numpy()).take(1)

    for file_path in files_path_data_plot:
        print('=' * 60)
        print(' PATH INFO '.center(60, '='))
        print('=' * 60)
        print(f'File Path: {file_path}')
        print()
        
        print('=' * 60)
        print(' SPLIT FILE PATH '.center(60, '='))
        print('=' * 60)
        split_file_path = tf.strings.split(file_path, os.path.sep)
        print(f'Split File Path: {split_file_path}')
        print()
        
        print('=' * 60)
        print(' INDEXED PATH '.center(60, '='))
        print('=' * 60)
        result = {value: f'Index -> {index}' for index, value in enumerate(split_file_path.numpy())}
        for key, value in result.items():
            print(f'{value}: {key}')
        print()

        print('=' * 60)
        print(f' KIND DATA INDEX: {kind_data} '.center(60, '='))
        print('=' * 60)
        index = tf.where(tf.equal(split_file_path, kind_data))[0][0]
        print(f'Index of "{kind_data}": {index}')
        print()

        print('=' * 60)
        print(' INDEX LABEL '.center(60, '='))
        print('=' * 60)
        index_label = index + 1
        print(f'Index Label: {index_label}')
        print()

        print('=' * 60)
        print(' LABEL '.center(60, '='))
        print('=' * 60)
        print(f'Label: {split_file_path[index_label]}')
        print()

        print('=' * 60)
        print(' FILE NAME '.center(60, '='))
        print('=' * 60)
        file_name = split_file_path[-1].numpy().decode('utf-8')
        print(f'File Name: {file_name}')
        print()

        print('=' * 60)
        print(' FILE EXTENSION '.center(60, '='))
        print('=' * 60)
        file_extension = os.path.splitext(file_name)[1]
        print(f'File Extension: {file_extension}')
        print()

        print('=' * 60)
        print(' FILE SIZE '.center(60, '='))
        print('=' * 60)
        file_size = os.path.getsize(file_path.numpy().decode('utf-8'))
        file_size = format_file_size(file_size, unit_file_size=unit_file_size)
        print(f'File Size: {file_size} {unit_file_size}')
        print()

# ==================================================== DATA TEST ====================================================
def show_test_files_path_info(files_path_data, is_random=False, unit_file_size='bytes'):
    
    if is_random:
        files_path_data_plot = files_path_data.take(1)
    else:
        files_path_data_plot = files_path_data.shuffle(buffer_size=files_path_data.cardinality().numpy()).take(1)

    for file_path in files_path_data_plot:
        print('=' * 60)
        print(' PATH INFO '.center(60, '='))
        print('=' * 60)
        print(f'File Path: {file_path}')
        print()
        
        print('=' * 60)
        print(' SPLIT FILE PATH '.center(60, '='))
        print('=' * 60)
        split_file_path = tf.strings.split(file_path, os.path.sep)
        print(f'Split File Path: {split_file_path}')
        print()
        
        print('=' * 60)
        print(' INDEXED PATH '.center(60, '='))
        print('=' * 60)
        result = {value: f'Index -> {index}' for index, value in enumerate(split_file_path.numpy())}
        for key, value in result.items():
            print(f'{value}: {key}')
        print()

        print('=' * 60)
        print(' FILE NAME '.center(60, '='))
        print('=' * 60)
        file_name = split_file_path[-1].numpy().decode('utf-8')
        print(f'File Name: {file_name}')
        print()

        print('=' * 60)
        print(' FILE EXTENSION '.center(60, '='))
        print('=' * 60)
        file_extension = os.path.splitext(file_name)[1]
        print(f'File Extension: {file_extension}')
        print()

        print('=' * 60)
        print(' FILE SIZE '.center(60, '='))
        print('=' * 60)
        file_size = os.path.getsize(file_path.numpy().decode('utf-8'))
        file_size = format_file_size(file_size, unit_file_size=unit_file_size)
        print(f'File Size: {file_size} {unit_file_size}')
        print()