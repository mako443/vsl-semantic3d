import numpy as np
import os
import pickle

'''
CARE: Only define here, import everywhere else
FOV estimated "by hand"
'''

FOV_W=64.0
FOV_H=44.8
IMAGE_WIDHT=1620
IMAGE_HEIGHT=1080

#ALL_SCENE_NAMES=['bildstein_station1_xyz_intensity_rgb','bildstein_station3_xyz_intensity_rgb','bildstein_station5_xyz_intensity_rgb','domfountain_station1_xyz_intensity_rgb','domfountain_station2_xyz_intensity_rgb','domfountain_station3_xyz_intensity_rgb','neugasse_station1_xyz_intensity_rgb','sg27_station1_intensity_rgb','sg27_station2_intensity_rgb','sg27_station4_intensity_rgb','sg27_station5_intensity_rgb','sg27_station9_intensity_rgb','sg28_station4_intensity_rgb','untermaederbrunnen_station1_xyz_intensity_rgb','untermaederbrunnen_station3_xyz_intensity_rgb']
#W/O SG_2
ALL_SCENE_NAMES=['bildstein_station1_xyz_intensity_rgb','bildstein_station3_xyz_intensity_rgb','bildstein_station5_xyz_intensity_rgb','domfountain_station1_xyz_intensity_rgb','domfountain_station2_xyz_intensity_rgb','domfountain_station3_xyz_intensity_rgb','neugasse_station1_xyz_intensity_rgb','sg27_station1_intensity_rgb','sg27_station4_intensity_rgb','sg27_station5_intensity_rgb','sg27_station9_intensity_rgb','sg28_station4_intensity_rgb','untermaederbrunnen_station1_xyz_intensity_rgb','untermaederbrunnen_station3_xyz_intensity_rgb']

class Pose:
    def __init__(self, scene_name, eye, right, up, forward, phi):
        self.scene_name=scene_name
        self.eye=eye
        self.right=right
        self.up=up
        self.forward=forward
        self.phi=phi
        self.I, self.E= None, None #Camera matrices, added during Open3D scene rendering

CLASSES_DICT={'unlabeled': 0, 'man-made terrain': 1, 'natural terrain': 2, 'high vegetation': 3, 'low vegetation': 4, 'buildings': 5, 'hard scape': 6, 'scanning artefacts': 7, 'cars': 8}
CLASSES_COLORS={'unlabeled': (255,255,255), 'man-made terrain': (60,30,30), 'natural terrain': (30,60,30), 'high vegetation': (120,255,120), 'low vegetation': (80,255,80), 'buildings': (255,255,0), 'hard scape': (0,255,255), 'scanning artefacts': (255,0,0), 'cars': (0,0,255)}
