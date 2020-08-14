import pickle
import semantic.utils
import numpy as np
import cv2
import os
from semantic.geometry import FOV_W,FOV_H,IMAGE_WIDHT,IMAGE_HEIGHT
from graphics.rendering import CLASSES_DICT, CLASSES_COLORS
from semantic.utils import draw_patches

'''
TODO
-Finish all pipeline w/ axis-aligned BBoxes, use min/max distance, evaluate
OR
-Finish pipeline via depth -> evaluate
'''

class Patch:
    __slots__ = ['label', 'bbox', 'depth', 'center']
    def __init__(self, class_label, bbox, depth):
        self.label=class_label
        self.bbox=np.int16(bbox) #As left,top,w,h
        self.depth=depth
        self.center=np.int16(( bbox[0]+0.5*bbox[2], bbox[1]+0.5*bbox[3] ))

def extract_patches(image_labels,image_depth):
    patches=[]

    #for class_label in ('high vegetation',):
    for class_label in ('man-made terrain','natural terrain','high vegetation','low vegetation','buildings','hard scape','cars'): #Disregard unknowns and artifacts
        #Extract the mask for the current label
        mask= image_labels == (CLASSES_COLORS[class_label][2],CLASSES_COLORS[class_label][1],CLASSES_COLORS[class_label][0]) #CARE: RGB<->BGR
        mask= np.all(mask,axis=2)   

        #TODO: merge split components? Probably simply by x or y overlap, small distance in other direction, same z | Dilate&Erode schwierig wegen Depth-Gaps... | Logic w/ merged bbox?
        cc_retval, cc_labels, cc_stats, cc_centroids = cv2.connectedComponentsWithStats(np.uint8(mask))
        mask=np.zeros_like(image_labels)
        for i in range(1,len(cc_centroids)):
            if cc_stats[i,cv2.CC_STAT_AREA]<4000: #OPTIONS: min-area
                continue
                
            patch_depth=np.mean(image_depth[ cc_labels==i, 0])
            patches.append(Patch(class_label, cc_stats[i,0:4],patch_depth))

    return patches

#Scene patches as { file_name: [patches] }
def gather_patches(base_path,scene_name):
    print('Gather scene',scene_name)
    dirpath_scene=os.path.join(base_path,scene_name)
    dirpath_labels=os.path.join(dirpath_scene,'lbl')
    dirpath_depth=os.path.join(dirpath_scene,'depth')
    dirpath_rgb=os.path.join(dirpath_scene,'rgb')
    assert os.path.isdir(dirpath_labels) and os.path.isdir(dirpath_depth)
    assert len(os.listdir(dirpath_depth)) == len(os.listdir(dirpath_labels)) 

    scene_patches={}
    for file_name in os.listdir(dirpath_labels):
        print(file_name)
        image_labels, image_depth= cv2.imread(os.path.join(dirpath_labels, file_name)), cv2.imread(os.path.join(dirpath_depth, file_name))
        view_patches=extract_patches(image_labels, image_depth)
        scene_patches[file_name]=view_patches
        
    return scene_patches

if __name__ == "__main__":
    base_path='data/pointcloud_images_3_2_depth'
    scene_name='sg27_station2_intensity_rgb'
    
    '''
    Patches extraction
    '''
    scene_patches=gather_patches(base_path,scene_name)
    pickle.dump( scene_patches, open(os.path.join(base_path, scene_name,'patches.pkl'), 'wb'))





