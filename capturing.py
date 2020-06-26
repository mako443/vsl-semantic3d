import numpy as np
import os
import pptk
import time
import sys

#play()/record() expect lookat-poses ✖ not possible: glitches

scene_config={}
scene_config['domfountain_station1_xyz_intensity_rgb']={
    #Nx3 array
    'points':np.array( [[-10.74167538,   7.32442617,  -0.50595516],
                        [ -4.53943491,   5.41002083,  -0.57776338],
                        [ -4.10716677,  -3.10575199,   0.08882368],
                        [-10.06188583,  -7.36892557,   0.50089562],
                        [-15.61357212,  -5.97002316,   0.73054999],
                        [-15.33024597,  13.69128513,  -0.38836336]]), 
    'point_size_rgb':0.025, #labels always double
}

#Care: repeat last point (or close by)
#Poses = lookat-positions
def points2poses(points, num_poses):
    points=np.array(points)
    assert points.shape[1]==3
    poses=np.zeros((num_poses,3))
    for i in range(3):
        poses[:,i]=np.interp(np.linspace(0,len(points)-1,num_poses), np.arange(len(points)), points[:,i])
    return poses

#INPUT: lookat-poses
#OUTPUT: lookat-poses, theta-angles
def capture_poses(viewer, path_rgb, path_labels, filepath_poses, poses, point_size_color, point_size_labels, num_angles):
    assert viewer.get('num_attributes')[0]==2
    assert os.path.exists(path_rgb) and os.path.exists(path_labels)

    viewer.set(show_grid=False, show_info=False, show_axis=False)
    viewer.set(bg_color=(0,0,0,1))
    viewer.set(bg_color_bottom=(0,0,0,1))
    viewer.set(bg_color_top=(0,0,0,1))

    poses_array=[]

    st=2.0
    #Render color images
    viewer.set(curr_attribute_id=0)
    viewer.set(point_size=point_size_color)
    for i_pose, pose in enumerate(poses):
        print(f'\r color pose {i_pose} of {len(poses)}',end='')
        viewer.set(lookat=pose)
        #time.sleep(st)
        for i_angle,phi in enumerate(np.linspace(np.pi, -np.pi,num_angles+1)[0:-1]): 
            viewer.set(phi=phi,theta=0.0,r=5.0)
            #time.sleep(st)
            #viewer.capture(os.path.join(path_rgb,f'{i_pose:03d}_{i_angle:02d}.png'))

            poses_array.append(np.hstack(( pose,phi,0.0,5.0 )))

    #Render label images
    viewer.set(curr_attribute_id=1)
    viewer.set(point_size=point_size_labels)
    for i_pose, pose in enumerate(poses):
        print(f'\r label pose {i_pose} of {len(poses)}',end='')
        viewer.set(lookat=pose)        
        #time.sleep(st)
        for i_angle,phi in enumerate(np.linspace(np.pi, -np.pi,num_angles+1)[0:-1]): 
            viewer.set(phi=phi,theta=0.0,r=5.0)
            #time.sleep(st)
            #viewer.capture(os.path.join(path_labels,f'{i_pose:03d}_{i_angle:02d}.png'))

    viewer.set(show_grid=True, show_info=True, show_axis=True)
    
    print('\n Saving poses...')
    np.array(poses_array).tofile(filepath_poses)


def visualize_poses(viewer,poses,ts=0.2):
    if poses.shape[1]==3:
        #poses=np.hstack(( poses,np.array([[0,0,5],]).repeat(len(poses),axis=0) ))
        poses=np.hstack(( poses,np.array([[np.pi/2,np.pi/2,30],]).repeat(len(poses),axis=0) ))
    viewer.play(poses,ts=ts*np.arange(len(poses)))


