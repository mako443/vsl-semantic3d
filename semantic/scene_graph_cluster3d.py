import numpy as np
import cv2
import os
import random
import pickle
<<<<<<< HEAD
import time
=======
>>>>>>> 16b4d214d5b9c165179bdb9467ed26ee28fd733c
from graphics.imports import CLASSES_DICT, CLASSES_COLORS, Pose, IMAGE_WIDHT, IMAGE_HEIGHT, COMBINED_SCENE_NAMES
from .imports import ClusteredObject, ViewObject
from .scene_graph_cluster3d_scoring import get_relationship_type, score_relationship_type

'''
Module to generate the view-objects from the clustered objects | Scene-Graph & text generation logic merges after view-objects are generated
'''

#Check if object is ahead of camera and enough of it's area is visible
def is_object_in_fov(obj : ClusteredObject):
    assert obj.rect_i is not None
    if obj.maxdist_i<0:
        return False

    MIN_AREA= 100*100 #OPTION: min-area in pixels²
    fov_points=np.array(( (0,0), (0,IMAGE_HEIGHT), (IMAGE_WIDHT,0), (IMAGE_WIDHT,IMAGE_HEIGHT) ))
    fov_rect=cv2.minAreaRect(fov_points.astype(np.float32))

    intersection_type, intersection_points=cv2.rotatedRectangleIntersection(fov_rect, obj.rect_i)
    if intersection_type is cv2.INTERSECT_NONE:
        return False

    intersection_rect=cv2.minAreaRect(intersection_points.astype(np.float32))
    intersection_area=np.prod(intersection_rect[1])
    return intersection_area>=MIN_AREA


#A 'stateless' check of object occlusion
#TODO: Rectangle-based occlusion checks do not work!
#New strategy: checking for a single occluder that covers most of objects area and is completely closer
#TODO: Occlusion check by re-projection of points_i back to lbl/dist image? | Occlusions one of the biggest problems?
def is_object_occluded(obj : ClusteredObject, visible_objects):
    #return False #Does not work / unreliable!
    #For each occluder in fov and closer:
    # occlusion_area=0.0

    for occluder in visible_objects:
        if obj==occluder: 
            continue

        if not obj.mindist_i-occluder.maxdist_i>2 and occluder.maxdist_i>0: #OPTION: distance difference
            continue

        #get the rotated-rect intersection
        intersection_type, intersection_points=cv2.rotatedRectangleIntersection(occluder.rect_i, obj.rect_i)

        # #if intersect -> add area (naive approach)
        # if intersection_points is not None:
        #     intersection_rect=cv2.minAreaRect(intersection_points)
        #     occlusion_area+=np.prod(intersection_rect[1])

        #Check for a single occluder that covers enough area (naive)
        if intersection_points is not None:
            intersection_rect=cv2.minAreaRect(intersection_points)
            if np.prod(intersection_rect[1])>0.8*obj.get_area(): #Option: occlusion area
                return True
    return False

    #return occlusion_area>=obj.get_area()

def create_view_objects(scene_objects, view_pose : Pose):
    I,E=view_pose.I, view_pose.E
    for o in scene_objects:
        o.project(I,E)

    fov_objects=[ obj for obj in scene_objects if obj.rect_i is not None and is_object_in_fov(obj) ]

    #print(f'Scene but not fov objects: {len(scene_objects) - len(fov_objects)}')
    visible_objects=[ obj for obj in fov_objects if not is_object_occluded(obj, fov_objects) ]
    #print(f'FoV but occluded objects: {len(fov_objects) - len(visible_objects)}')
    view_objects=[ ViewObject.from_clustered_object(obj) for obj in visible_objects ]

    return view_objects

if __name__ == "__main__":
    scene_name='neugasse_station1_xyz_intensity_rgb'
    scene_objects=pickle.load( open('data/numpy_merged/'+scene_name+'.objects.pkl', 'rb'))
    poses_rendered=pickle.load( open( os.path.join('data','pointcloud_images_o3d_merged',scene_name,'poses_rendered.pkl'), 'rb'))
    
    #file_name='029.png'
    file_name=np.random.choice(list(poses_rendered.keys()))
    pose=poses_rendered[file_name]
    img=cv2.imread( os.path.join('data','pointcloud_images_o3d_merged',scene_name,'rgb', file_name) )

    view_objects=create_view_objects(scene_objects,pose)
    #view_objects=[v for v in view_objects if v.label in ('cars','buildings', 'low vegetation')]
    print('view objects',len(view_objects), 'for ',file_name)

    for v in view_objects:
        if "terrain" in v.label or True:
            v.draw_on_image(img)

    cv2.imshow("",img)
    cv2.waitKey()

    quit()

    '''
    Data creation: View objects from clustered objects
    '''
    base_dir='data/pointcloud_images_o3d_merged/'
    for i_scene_name,scene_name in enumerate(COMBINED_SCENE_NAMES):
<<<<<<< HEAD
    #for scene_name in ('sg27_station5_intensity_rgb',)
=======
    #for scene_name in ('sg27_station5_intensity_rgb',):
>>>>>>> 16b4d214d5b9c165179bdb9467ed26ee28fd733c
        print()
        print("Scene: ",scene_name)
        scene_objects=pickle.load( open('data/numpy_merged/'+scene_name+'.objects.pkl', 'rb'))
        poses_rendered=pickle.load( open( os.path.join(base_dir,scene_name,'poses_rendered.pkl'), 'rb'))

        scene_view_objects={}
        total_view_objects=0
        for i_file,file_name in enumerate(poses_rendered.keys()):
            pose=poses_rendered[file_name]
            view_objects=create_view_objects(scene_objects,pose)
            total_view_objects+=len(view_objects)
            scene_view_objects[file_name]=view_objects
            #print(f'\r file {i_file} of {len(poses_rendered)}',end='')

        print()
        print('Saving view objects...', total_view_objects/len(scene_view_objects),'view objects on average')
        pickle.dump( scene_view_objects, open(os.path.join(base_dir,scene_name,'view_objects.pkl'), 'wb'))

    quit()  