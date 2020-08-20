import numpy as np
import os
import pptk
import time
import sys
import pickle
from .imports import Pose
from main import load_files, view_pptk, resize_window

'''
Module to find the poses and save them
'''
scene_config={}
scene_config['domfountain_station1_xyz_intensity_rgb']={
    #Nx3 array
    'points':np.array( [[-10.74167538,   10.32442617,  -0.50595516],
                        [ -4.53943491,   5.41002083,  -0.57776338],
                        [ -4.10716677,  -3.10575199,   0.08882368],
                        [-10.06188583,  -7.36892557,   0.50089562],
                        [-15.61357212,  -5.97002316,   0.73054999],
                        [-15.33024597,  16.69128513,  -0.38836336]]), 
    'point_size_rgb':0.025, #labels always double
}
scene_config['sg27_station2_intensity_rgb']={
    #Nx3 array
    'points':np.array( [[ -2.28320765, -13.30841255,   0.65572155],
                        [ 14.10788918, -32.98677063,  -1.28365839],
                        [ 26.90045547, -23.60473442,  -1.01840901],
                        [ 23.01133728,   0.17587358,  -1.40091133],
                        [ -1.34517264,  10.82786083,   0.17833348]]), 
    'point_size_rgb':0.025, #labels always double
}
scene_config['untermaederbrunnen_station1_xyz_intensity_rgb']={
    #Nx3 array
    'points':np.array([[  2.03187799,  -2.71359539,  -0.16611724],
                        [  1.21940982,   5.60008621,  -0.61616558],
                        [-18.17295837,  17.23288536,   0.42189693],
                        [-14.96191025,   6.38012838,  -0.22417469],
                        [ -1.3910284 ,   1.32536995,  -0.45402691]]),
    'point_size_rgb':0.025, #labels always double
}
scene_config['neugasse_station1_xyz_intensity_rgb']={
    #Nx3 array
    'points':np.array([[ 1.1074542e+01,  1.0536696e+00, -4.0069711e-01],
                        [-3.3289959e+00,  1.0078371e-02,  1.9622801e-01],
                        [-9.9907360e+00, -3.6623685e+00, -3.0740970e-01],
                        [-3.0083437e+00, -8.5878935e+00, -9.0476435e-01],
                        [ 1.7781239e+00, -3.6863661e+00, -9.0476412e-01]]), 
    'point_size_rgb':0.025, #labels always double
}

def interpolate_points(points_in, num_points):
    points_in=np.array(points_in)
    assert points_in.shape[1]==3

    points=np.zeros((num_points,3))
    for i in range(3):
        points[:,i]=np.interp(np.linspace(0,len(points_in)-1,num_points), np.arange(len(points_in)), points_in[:,i])
    return points

def calculate_poses(viewer, scene_name, points, num_angles, visualize=False):
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

if __name__ == "__main__":
    '''
    View pptk -> write config -> interpolate&save poses
    '''    
    scene_name='domfountain_station1_xyz_intensity_rgb'
    output_path_poses=os.path.join('data','pointcloud_images_o3d',scene_name,'poses.pkl')

    viewer=view_pptk('data/numpy/'+scene_name, remove_artifacts=True, remove_unlabeled=True, max_points=int(10e6))
    resize_window()

    points=scene_config[scene_name]['points']
    points=interpolate_points(points,20)

    poses=calculate_poses(viewer, scene_name, points, num_angles=10)
    print('num poses:',len(poses))
    pickle.dump( poses, open(output_path_poses, 'wb') )
    print('Poses saved!', output_path_poses)