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

'''
TODO

Overlaps: domfountain, bildstein, 
'''
scene_config={}
scene_config['bildstein_station1_xyz_intensity_rgb']={
    #Nx3 array
    'points':np.array( [[  6.89986515, -23.8486042 ,   2.24763012],
                        [ 14.23859787,  -6.84959984,   0.59534997],
                        [-12.54539108,   8.38919926,  -1.56900454],
                        [-12.71178436,  25.77069092,  -1.69532895],
                        [ 12.06625557,  28.25757217,  -1.58685946],
                        [ 10.43738079,   6.53896713,   0.22520876]]), 
    'point_size_rgb':0.025, #labels always double
}
scene_config['bildstein_station3_xyz_intensity_rgb']={
    #Nx3 array
    'points':np.array( [[-13.66415501,  -1.87494004,   1.20972967],
                        [-15.5337286 ,  18.76299858,   2.06882811],
                        [  2.10001707,  25.09252548,   2.41423416],
                        [  7.10345364,  -7.98233128,   0.36108246],
                        [ 16.39338684, -35.32163239,  -2.28681469]]), 
    'point_size_rgb':0.025, #labels always double
}
scene_config['bildstein_station5_xyz_intensity_rgb']={
    #Nx3 array
    'points':np.array( [[-19.88222122,  34.78226089,   1.30118823],
                        [ 17.61147308,   6.93702793,  -2.1327517 ],
                        [  7.25930929, -15.78915977,  -1.98583078],
                        [-12.99044991, -32.33332443,  -1.03720355],
                        [-31.48099136, -48.11365891,   1.69673967]]), 
    'point_size_rgb':0.025, #labels always double
}
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
scene_config['domfountain_station2_xyz_intensity_rgb']={
    #Nx3 array
    'points':np.array( [[-29.57394791,   9.05232334,   0.47919855],
                        [  4.34464264,  -3.3333323 ,   0.15014482],
                        [ -0.56595898, -19.08597183,  -0.21233426],
                        [ -1.47728717, -34.15919495,   0.49576887],
                        [ 11.51516724, -12.17238903,  -0.28104484],
                        [ 26.30233955, -10.83860207,  -0.11856288]]), 
    'point_size_rgb':0.025, #labels always double
}
scene_config['domfountain_station3_xyz_intensity_rgb']={
    #Nx3 array
    'points':np.array( [[ -7.10327578,  33.63069534,  -0.16973461],
                        [  0.32602367,   7.58825922,  -0.12918571],
                        [  4.26744795, -14.16171265,   0.35794488],
                        [ 23.29409027, -36.65826797,   0.81396568]]), 
    'point_size_rgb':0.025, #labels always double
}
scene_config['sg27_station2_intensity_rgb']={
    #Nx3 array
    'points':np.array( [[ -2.28320765, -13.30841255,   0.65572155],
                        [ 14.10788918, -32.98677063,  -1.28365839],
                        [ 26.90045547, -23.60473442,  -1.01840901],
                        [ 23.01133728,  10.0      ,  -1.40091133],
                        [ -1.34517264,  10.82786083,   0.17833348]]), 
    'point_size_rgb':0.025, #labels always double
}
# scene_config['untermaederbrunnen_station1_xyz_intensity_rgb']={
#     #Nx3 array
#     'points':np.array([[  2.03187799,  -2.71359539,  -0.16611724],
#                         [  1.21940982,   5.60008621,  -0.61616558],
#                         [-18.17295837,  17.23288536,   0.42189693],
#                         [-14.96191025,   6.38012838,  -0.22417469],
#                         [ -1.3910284 ,   1.32536995,  -0.45402691]]),
#     'point_size_rgb':0.025, #labels always double
# }

#TODO: Non-circle ok?
scene_config['untermaederbrunnen_station1_xyz_intensity_rgb']={
    #Nx3 array
    'points':np.array([[-8.15726089e+00, -1.17434473e+01, -2.00530887e-03],
                        [9.46222305, 3.13939857, 0.07745753],
                        [-7.56795025,  6.47402334, -0.09737678],
                        [-19.66889381,  13.40221691,  -0.38015223]]),
    'point_size_rgb':0.025, #labels always double
}

#CARE: z changed to higher
scene_config['neugasse_station1_xyz_intensity_rgb']={
    #Nx3 array
    'points':np.array([[ 1.1074542e+01,  1.0536696e+00,  1.0],
                        [-3.3289959e+00,  1.0078371e-02, 1.0],
                        [-9.9907360e+00, -3.6623685e+00, 1.0],
                        [-3.0083437e+00, -8.5878935e+00, 1.0],
                        [ 1.7781239e+00, -3.6863661e+00, 1.0]]), 
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
    scene_name='bildstein_station5_xyz_intensity_rgb'
    output_path_poses=os.path.join('data','pointcloud_images_o3d',scene_name,'poses.pkl')

    viewer=view_pptk('data/numpy/'+scene_name, remove_artifacts=True, remove_unlabeled=True, max_points=int(10e6))
    resize_window()
    quit()

    points=scene_config[scene_name]['points']
    points=interpolate_points(points,20)
    # visualize_points(viewer,points)
    # quit()

    poses=calculate_poses(viewer, scene_name, points, num_angles=10)
    print('num poses:',len(poses))
    pickle.dump( poses, open(output_path_poses, 'wb') )
    print('Poses saved!', output_path_poses)