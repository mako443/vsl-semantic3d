import numpy as np
import os
import pptk
import time
import sys
import pickle
from .imports import Pose
from main import load_files, view_pptk, resize_window
from graphics.imports import CLASSES_DICT, CLASSES_COLORS, Pose, IMAGE_WIDHT, IMAGE_HEIGHT, COMBINED_SCENE_NAMES
import graphics.poses_train
import graphics.poses_test

'''
Module to find the poses and save them

TODO:
-assert that the poses files are there, add comment
'''

def interpolate_points(points_in, num_points):
    points_in=np.array(points_in)
    assert points_in.shape[1]==3

    if len(points_in)==num_points:
        return points_in

    points=np.zeros((num_points,3))
    for i in range(3):
        points[:,i]=np.interp(np.linspace(0,len(points_in)-1,num_points), np.arange(len(points_in)), points_in[:,i])
    return points

def calculate_poses(viewer, scene_name, points, num_angles, visualize=False):
    time.sleep(1.0)
    poses=[]
    for point in points:
        viewer.set(lookat=point)
        for i_angle,phi in enumerate(np.linspace(np.pi, -np.pi,num_angles+1)[0:-1]): 
            viewer.set(phi=phi,theta=0.0,r=0.0)
            if visualize:
                time.sleep(0.5)

            poses.append( Pose(scene_name, viewer.get('eye'), viewer.get('right'), viewer.get('up'), viewer.get('view'), phi) )

    return poses

def visualize_points(viewer,points,ts=0.2):
    if points.shape[1]==3:
        points=np.hstack(( points,np.array([[np.pi/2,np.pi/2,30],]).repeat(len(points),axis=0) ))
    viewer.play(points,ts=ts*np.arange(len(points)))

def add_viewpoints_to_pointcloud(xyz, rgba, labels_rgba, points, split):
    assert split in ('train','test')
    color= (255,0,0,255) if split=='train' else (0,0,255,255) #Red:train, Blue: test
    xyz_base=np.array(np.meshgrid(np.linspace(-0.5,0.5,5),np.linspace(-0.5,0.5,5),np.linspace(-0.5,0.5,5))).T.reshape((-1,3))
    for point in points:
        xyz=np.vstack(( xyz, xyz_base+point ))
        rgba=np.vstack(( rgba, color*np.ones(( len(xyz_base),4 ), dtype=np.uint8) ))
        labels_rgba=np.vstack(( labels_rgba, color*np.ones(( len(xyz_base),4 ), dtype=np.uint8) ))

    return xyz, rgba, labels_rgba
        

if __name__ == "__main__":
    '''
    View pptk -> write config -> interpolate&save poses
    '''    
    # scene_name='sg27_station1_intensity_rgb'
    # output_path_poses=os.path.join('data','pointcloud_images_o3d_merged',scene_name,'poses.pkl')

    # #viewer=view_pptk('data/numpy_merged/'+scene_name, remove_artifacts=True, remove_unlabeled=True, max_points=int(15e6))

    # #num_points=scene_config[scene_name]['num_points']
    # #points=scene_config[scene_name]['points']

    # num_points_train=graphics.poses_train.scene_config[scene_name]['num_points']
    # points_train=graphics.poses_train.scene_config[scene_name]['points']
    # points_train=interpolate_points(points_train,num_points_train)

    # num_points_test=graphics.poses_test.scene_config[scene_name]['num_points']
    # points_test=graphics.poses_test.scene_config[scene_name]['points']
    # if len(points_test)>0: points_test=interpolate_points(points_test,num_points_test)    

    # xyz, rgba, labels_rgba=load_files('data/numpy_merged/'+scene_name, remove_artifacts=True, remove_unlabeled=True, max_points=int(15e6))
    # xyz, rgba, labels_rgba=add_viewpoints_to_pointcloud(xyz, rgba, labels_rgba, points_train, 'train')
    # xyz, rgba, labels_rgba=add_viewpoints_to_pointcloud(xyz, rgba, labels_rgba, points_test , 'test')

    # viewer=pptk.viewer(xyz)
    # viewer.attributes(rgba.astype(np.float32)/255.0,labels_rgba.astype(np.float32)/255.0)
    # viewer.set(point_size=0.025)

    # resize_window()

    # viewer.capture(f'split_{scene_name}.png')    

    # points=interpolate_points(points,num_points)
    # visualize_points(viewer,points)
    # quit()

    '''
    Data creation: calculate poses
    '''
    base_path='data/pointcloud_images_o3d_merged'    
    for scene_name in COMBINED_SCENE_NAMES:
        #for split in ('train','test',):
        for split in ('test',):
            print(f'\n\n Scene: {scene_name} split: {split}')
            if split=='train':
                num_points=graphics.poses_train.scene_config[scene_name]['num_points']
                points=graphics.poses_train.scene_config[scene_name]['points']
                num_angles=10
            if split=='test':
                num_points=graphics.poses_test.scene_config[scene_name]['num_points']
                points=graphics.poses_test.scene_config[scene_name]['points']
                num_angles=6                

            points=interpolate_points(points, num_points)
            output_path_poses=os.path.join('data','pointcloud_images_o3d_merged',split,scene_name,'poses.pkl')
            print(f'Output path poses: {output_path_poses}')

            #Using viewer-math to set the poses | viewer get's stuck after too many requests / naive work-around
            poses=[]
            for start_idx in range(0,len(points),10):
                viewer=view_pptk('data/numpy_merged/'+scene_name, remove_artifacts=True, remove_unlabeled=True, max_points=int(15e6))
                resize_window()
                poses.extend( calculate_poses(viewer, scene_name, points[start_idx:start_idx+10], num_angles=num_angles) )
                viewer.close()

            #poses=calculate_poses(viewer, scene_name, points, num_angles=10)
            print('num poses:',len(poses))
            pickle.dump( poses, open(output_path_poses, 'wb') )
            print('Poses saved!', output_path_poses)
