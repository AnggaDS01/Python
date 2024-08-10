import shutil
main_folder_path = './Assets/Datasets/SIBI dataset/Test/'
subfolders = [f.path for f in os.scandir(main_folder_path) if f.is_dir()]
for subfolder in subfolders:
    for file_name in os.listdir(subfolder):
        file_path = os.path.join(subfolder, file_name)
        if os.path.isfile(file_path):
            shutil.move(file_path, main_folder_path)
    os.rmdir(subfolder)