import numpy as np
import os
import pickle
import open3d
import time
import logging
from graphics.imports import ALL_SCENE_NAMES

VOXEL_SIZE=0.02 #OPTION: Voxel-size, the base voxel-size for the first down-sampling

#TODO: pull SG station 2
#TODO: voxel-down from start on ok? Smaller voxels? Visually: Looks even better, Clustering: Looks good too, possibly re-tweak params
def convert_downsample(filepath_in_points, filepath_in_labels, filepath_xyz, filepath_rgb, filepath_labels, max_points=int(100e6)):
    assert os.path.isfile(filepath_in_points)
    assert os.path.isfile(filepath_in_labels)
    assert not os.path.isfile(filepath_xyz)
    assert not os.path.isfile(filepath_rgb)
    assert not os.path.isfile(filepath_labels)    

    #Load points from .txt
    logging.info('\n\n')
    logging.info('loading points for',filepath_in_points )
    points=np.loadtxt(filepath_in_points, delimiter=' ', dtype=np.float32)
    logging.info(f'loaded {len(points)} points')

    #Load labels from .txt
    labels=np.loadtxt(filepath_in_labels, dtype=np.uint8).flatten()
    assert len(labels)==len(points) 

    #Perform voxel-downsampling to reduce all points
    point_cloud=open3d.geometry.PointCloud()
    point_cloud.points=open3d.utility.Vector3dVector(points[:,0:3].copy())
    _,_,indices_list=point_cloud.voxel_down_sample_and_trace(VOXEL_SIZE,point_cloud.get_min_bound(), point_cloud.get_max_bound()) 
    logging.info(f'Reduced to {len(indices_list)} points after voxel-down')
    
    #Not vectorized but seems fast enough, CARE: first-index color sampling (not averaging)
    indices=np.array([ vec[0] for vec in indices_list ])

    #Reduce points by simple removal
    if len(indices)>max_points:
        step=int(np.ceil(len(indices)/max_points))
        assert step>1
        indices=indices[::step]
        logging.debug(f'Reduces to {len(indices)} points by simple removal')
    else:
        logging.debug('No removal necessary')

    #Save xyz, rgb and labels in separate numpy files
    np.save(open(filepath_xyz,'wb'), np.float32(points[indices,0:3]))
    np.save(open(filepath_rgb,'wb'), np.uint8(  points[indices,4:7]))
    np.save(open(filepath_labels,'wb'), np.uint8(  labels[indices]))

if __name__ == "__main__":
    logging.basicConfig(filename='log'+os.path.basename(__file__)+'.log', level=logging.DEBUG)

    #for scene_name in ALL_SCENE_NAMES:
    for scene_name in ['sg27_station2_intensity_rgb','sg27_station4_intensity_rgb','sg27_station5_intensity_rgb','sg27_station9_intensity_rgb','sg28_station4_intensity_rgb','untermaederbrunnen_station1_xyz_intensity_rgb','untermaederbrunnen_station3_xyz_intensity_rgb']:
        convert_downsample(os.path.join('data','raw',scene_name+'.txt'),
                           os.path.join('data','labels',scene_name+'.labels'),
                           os.path.join('data','numpy',scene_name+'.xyz.npy'),
                           os.path.join('data','numpy',scene_name+'.rgb.npy'),
                           os.path.join('data','numpy',scene_name+'.lbl.npy'))


