import numpy as np
import cv2
import os
from main import load_files, classes_dict
import open3d
import pptk
from sklearn.cluster import SpectralClusteringy

#Point-cloud clustering and hidden points: http://www.open3d.org/docs/release/tutorial/Basic/pointcloud.html -> Doesn't seem to work ✖
#Simple depth-map: compute distance from eye for all points, color them accordingly (scaled to max)
#Occlusion check: if 2D-overlap >0.5 of object, check if other object is closer (simple bouding boxes 2D)
#Artifacts try #2: cluster, then all inside 3d bboxes 

'''
TODO
New strategy: search for small/big enough 2D blobs (dep. on class), describe 3D-relations using depth (small-small, small-big)
check for congruent depth (small only)?
Boden nur erwähnen (left, right, mid, across)

Fit rotated rects for in-front-of / behind?
'''



if __name__ == "__main__":
    pass