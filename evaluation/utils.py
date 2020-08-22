import pickle
import numpy as np
import cv2
import os

def load_all_poses_and_patches(base_path):
    scene_names=[ os.path.listdir(base_path) ]
    all_poses={}
    all_patches={}

    for scene_name in scene_names:
        