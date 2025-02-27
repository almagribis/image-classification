import os, sys, glob
path_root = os.path.abspath('..')
sys.path.insert(0, path_root)

def images_name():
    """get image name list"""
    return sorted(os.listdir(os.path.join(path_root, "sample")))

def images_path():
    """get image path list"""

    return sorted(glob.glob(os.path.join(path_root, "sample", "*")))