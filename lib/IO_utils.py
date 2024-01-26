import os


def create_dir_if_not_existent(directory_path, is_subdir=False):
    if not os.path.isdir(directory_path):
        if is_subdir:
            os.system('mkdir -p ' + directory_path)
        else:
            os.mkdir(directory_path)
