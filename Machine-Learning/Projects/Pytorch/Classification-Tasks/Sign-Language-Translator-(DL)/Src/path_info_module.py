import os
import numpy as np

def get_random_samples_file_info(file_paths):
    random_index = np.random.randint(len(file_paths))
    selected_file_path = file_paths[random_index]
    
    split_path = selected_file_path.split(os.path.sep)
    dataset_type = split_path[-3]
    indexed_split_path = {value: f'Index -> {index}' for index, value in enumerate(split_path)}
    
    dataset_type_index = np.where(np.equal(split_path, dataset_type))[0][0]
    label_index = dataset_type_index + 1
    file_name = split_path[-1]
    file_extension = os.path.splitext(file_name)[1]
    file_size_kb = np.round(os.path.getsize(selected_file_path) / 1024, 4)
    
    return {
        "file_path": selected_file_path,
        "split_path": split_path,
        "indexed_split_path": indexed_split_path,
        "dataset_type": dataset_type,
        "dataset_type_index": dataset_type_index,
        "label_index": label_index,
        "label": split_path[label_index],
        "file_name": file_name,
        "file_extension": file_extension,
        "file_size_kb": file_size_kb
    }

def get_random_test_file_info(file_paths):
    random_index = np.random.randint(len(file_paths))
    selected_file_path = file_paths[random_index]
    
    split_path = selected_file_path.split(os.path.sep)
    dataset_type = split_path[-2]
    indexed_split_path = {value: f'Index -> {index}' for index, value in enumerate(split_path)}
    
    dataset_type_index = np.where(np.equal(split_path, dataset_type))[0][0]
    file_name = split_path[-1]
    file_extension = os.path.splitext(file_name)[1]
    file_size_kb = np.round(os.path.getsize(selected_file_path) / 1024, 4)
    
    return {
        "file_path": selected_file_path,
        "split_path": split_path,
        "indexed_split_path": indexed_split_path,
        "dataset_type": dataset_type,
        "dataset_type_index": dataset_type_index,
        "file_name": file_name,
        "file_extension": file_extension,
        "file_size_kb": file_size_kb
    }

def print_file_info(file_info, is_test=False):
    print('=' * 60)
    print(' PATH INFO '.center(60, '='))
    print('=' * 60)
    print(f'File Path: {file_info["file_path"]}')
    print('=' * 60)
    print()

    print('=' * 60)
    print(' SPLIT FILE PATH '.center(60, '='))
    print('=' * 60)
    print(f'Split File Path: {file_info["split_path"]}')
    print('=' * 60)
    print()

    print('=' * 60)
    print(' INDEXED PATH '.center(60, '='))
    print('=' * 60)
    for key, value in file_info["indexed_split_path"].items():
        print(f'{value}: {key}')
    print('=' * 60)
    print()

    print('=' * 60)
    print(f' DATASET TYPE INDEX: {file_info["dataset_type"]} '.center(60, '='))
    print('=' * 60)
    print(f'Index of "{file_info["dataset_type"]}": {file_info["dataset_type_index"]}')
    print('=' * 60)
    print()

    if not is_test:
        print('=' * 60)
        print(' INDEX LABEL '.center(60, '='))
        print('=' * 60)
        print(f'Label Index: {file_info["label_index"]}')
        print('=' * 60)
        print()

        print('=' * 60)
        print(' LABEL '.center(60, '='))
        print('=' * 60)
        print(f'Label: {file_info["label"]}')
        print('=' * 60)
        print()

    print('=' * 60)
    print(' FILE NAME '.center(60, '='))
    print('=' * 60)
    print(f'File Name: {file_info["file_name"]}')
    print('=' * 60)
    print()

    print('=' * 60)
    print(' FILE EXTENSION '.center(60, '='))
    print('=' * 60)
    print(f'File Extension: {file_info["file_extension"]}')
    print('=' * 60)
    print()

    print('=' * 60)
    print(' FILE SIZE '.center(60, '='))
    print('=' * 60)
    print(f'File Size: {file_info["file_size_kb"]} kb')
    print('=' * 60)
    print()
