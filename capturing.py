import numpy as np
import os
import pptk
import time
import sys
import pickle

#play()/record() expect lookat-poses ✖ not possible: glitches

# scene_config={}
# scene_config['domfountain_station1_xyz_intensity_rgb']={
#     #Nx3 array
#     'points':np.array( [[-10.74167538,   7.32442617,  -0.50595516],
#                         [ -4.53943491,   5.41002083,  -0.57776338],
#                         [ -4.10716677,  -3.10575199,   0.08882368],
#                         [-10.06188583,  -7.36892557,   0.50089562],
#                         [-15.61357212,  -5.97002316,   0.73054999],
#                         [-15.33024597,  13.69128513,  -0.38836336]]), 
#     'point_size_rgb':0.025, #labels always double
# }
# scene_config['sg27_station2_intensity_rgb']={
#     #Nx3 array
#     'points':np.array( [[ -2.28320765, -13.30841255,   0.65572155],
#                         [ 14.10788918, -32.98677063,  -1.28365839],
#                         [ 26.90045547, -23.60473442,  -1.01840901],
#                         [ 23.01133728,   0.17587358,  -1.40091133],
#                         [ -1.34517264,  10.82786083,   0.17833348]]), 
#     'point_size_rgb':0.025, #labels always double
# }
# scene_config['untermaederbrunnen_station1_xyz_intensity_rgb']={
#     #Nx3 array
#     'points':np.array([[  2.03187799,  -2.71359539,  -0.16611724],
#                         [  1.21940982,   5.60008621,  -0.61616558],
#                         [-18.17295837,  17.23288536,   0.42189693],
#                         [-14.96191025,   6.38012838,  -0.22417469],
#                         [ -1.3910284 ,   1.32536995,  -0.45402691]]),
#     'point_size_rgb':0.025, #labels always double
# }
# scene_config['neugasse_station1_xyz_intensity_rgb']={
#     #Nx3 array
#     'points':np.array([[ 1.1074542e+01,  1.0536696e+00, -4.0069711e-01],
#                         [-3.3289959e+00,  1.0078371e-02,  1.9622801e-01],
#                         [-9.9907360e+00, -3.6623685e+00, -3.0740970e-01],
#                         [-3.0083437e+00, -8.5878935e+00, -9.0476435e-01],
#                         [ 1.7781239e+00, -3.6863661e+00, -9.0476412e-01]]), 
#     'point_size_rgb':0.025, #labels always double
# }

#Care: repeat last point (or close by)
#Poses = lookat-positions
def points2poses(points, num_poses):
    points=np.array(points)
    assert points.shape[1]==3
    poses=np.zeros((num_poses,3))
    for i in range(3):
        poses[:,i]=np.interp(np.linspace(0,len(points)-1,num_poses), np.arange(len(points)), points[:,i])
    return poses

#TODO: improve from euclidean distance to view-dir. distance (inner product w/ max-length view-vector?)
#CARE: set viewer.color_map([[0,0,0],[0,0,1]], scale=[0,1]), returning relative distances!
def compute_depth_attribute(xyz, eye):
    depth=xyz-eye.astype(np.float32) #Keep all as float32 to reduce memory
    assert len(depth.shape)==2
    depth=np.linalg.norm(depth,axis=1)
    depth=depth/np.max(depth)
    return depth

#INPUT: lookat-poses
#OUTPUT: lookat-poses, theta-angles
def capture_poses(viewer, out_filepath_poses, poses, point_size_color, point_size_labels, path_rgb=None, path_labels=None, path_depth=None, xyz=None, num_angles=12):
    assert viewer.get('num_attributes')[0]==2

    R_VALUE=0.0

    if path_depth is None or xyz is None:
        print('capture_poses(): not rendering depth')

    viewer.set(show_grid=False, show_info=False, show_axis=False)
    viewer.set(bg_color=(0,0,0,1))
    viewer.set(bg_color_bottom=(0,0,0,1))
    viewer.set(bg_color_top=(0,0,0,1))

    st=2.0
    #CARE: Setting poses_dict redundantly for every type

    #Render color images
    if path_rgb is not None:
        print('\nRendering rgb...')
        assert os.path.isdir(path_rgb)
        poses_dict={}

        viewer.set(curr_attribute_id=0)
        viewer.set(point_size=point_size_color)
        for i_pose, pose in enumerate(poses):
            print(f'\r color pose {i_pose} of {len(poses)}',end='')    
            viewer.set(lookat=pose)
            time.sleep(st)
            for i_angle,phi in enumerate(np.linspace(np.pi, -np.pi,num_angles+1)[0:-1]): 
                viewer.set(phi=phi,theta=0.0,r=R_VALUE)
                time.sleep(st)
                file_name=f'{i_pose:03d}_{i_angle:02d}.png'
                target_path=os.path.join(path_rgb,file_name)
                viewer.capture(target_path)
                poses_dict[file_name]=np.hstack(( pose,phi,0.0,R_VALUE ))

    time.sleep(st) #sleep again to finish capturing

    #Render label images
    if path_labels is not None:
        print('\nRendering labels...')
        assert os.path.isdir(path_labels)
        poses_dict={}

        viewer.set(curr_attribute_id=1)
        viewer.set(point_size=point_size_labels)
        for i_pose, pose in enumerate(poses):
            print(f'\r label pose {i_pose} of {len(poses)}',end='')
            viewer.set(lookat=pose)        
            time.sleep(st)
            for i_angle,phi in enumerate(np.linspace(np.pi, -np.pi,num_angles+1)[0:-1]): 
                viewer.set(phi=phi,theta=0.0,r=R_VALUE)
                time.sleep(st)
                file_name=f'{i_pose:03d}_{i_angle:02d}.png'
                target_path=os.path.join(path_labels,file_name)
                viewer.capture(target_path)
                poses_dict[file_name]=np.hstack(( pose,phi,0.0,R_VALUE ))

    time.sleep(st) #sleep again to finish capturing

    #Render depth images
    if path_depth is not None and xyz is not None:
        print('\nRendering depth...')
        assert os.path.isdir(path_depth)
        poses_dict={}

        viewer.color_map([[0,0,0],[0,0,1]], scale=[0,1])
        for i_pose, pose in enumerate(poses):
            print(f'\r depth pose {i_pose} of {len(poses)}',end='')  
            viewer.set(lookat=pose)
            time.sleep(st)
            for i_angle,phi in enumerate(np.linspace(np.pi, -np.pi,num_angles+1)[0:-1]): 
                viewer.set(phi=phi,theta=0.0,r=R_VALUE)   
                depth=compute_depth_attribute(xyz, viewer.get('eye'))
                viewer.attributes(depth + 0.01) #Add 0.01 depth to distinguish to black
                time.sleep(st)
                file_name=f'{i_pose:03d}_{i_angle:02d}.png'
                target_path=os.path.join(path_depth,file_name)
                viewer.capture(target_path)
                poses_dict[file_name]=np.hstack(( pose,phi,0.0,R_VALUE ))


    time.sleep(st) #sleep again to finish capturing

    viewer.set(show_grid=True, show_info=True, show_axis=True)
    
    print('\n Saving poses...')
    pickle.dump( poses_dict, open(out_filepath_poses,'wb') )

#TODO: also visualize with red points, very fine interp. + noise if necessary
def visualize_poses(viewer,poses,ts=0.2):
    if poses.shape[1]==3:
        #poses=np.hstack(( poses,np.array([[0,0,5],]).repeat(len(poses),axis=0) ))
        poses=np.hstack(( poses,np.array([[np.pi/2,np.pi/2,30],]).repeat(len(poses),axis=0) ))
    viewer.play(poses,ts=ts*np.arange(len(poses)))


