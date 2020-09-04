import numpy as np
import os
import pickle
import open3d
import time
from graphics.imports import ALL_SCENE_NAMES

VOXEL_SIZE=0.02 #OPTION: Voxel-size, the base voxel-size for the first down-sampling

def load_safely(path):
    points=np.array([],dtype=np.float32).reshape((0,7))
    iter=0
    CHUNK_SIZE=50000000
    while True:
        print('Loading chunk', iter)
        try:
            chunk=np.loadtxt(path, delimiter=' ', dtype=np.float32, skiprows=iter*CHUNK_SIZE, max_rows=CHUNK_SIZE)
            points=np.vstack(( points,chunk ))
        except:
            break
        iter+=1

    return points

#TODO: pull SG station 2
#TODO: voxel-down from start on ok? Smaller voxels? Visually: Looks even better, Clustering: Looks good too, possibly re-tweak params
def convert_downsample(filepath_in_points, filepath_in_labels, filepath_xyz, filepath_rgb, filepath_labels, max_points=int(100e6)):
    assert os.path.isfile(filepath_in_points)
    assert os.path.isfile(filepath_in_labels)
    assert not os.path.isfile(filepath_xyz)
    assert not os.path.isfile(filepath_rgb)
    assert not os.path.isfile(filepath_labels)    

    #Load points from .txt
    print('\n\n')
    print('loading points for',filepath_in_points )
    if 'sg27_station2_intensity_rgb' in filepath_in_points: #CARE: have to treat the big scene differently
        points=np.load(open('data/raw/sg27_station2_intensity_rgb.npy','rb'))
    else:
        points=np.loadtxt(filepath_in_points, delimiter=' ', dtype=np.float32)
    print(f'loaded {len(points)} points')

    #Load labels from .txt
    labels=np.loadtxt(filepath_in_labels, dtype=np.uint8).flatten()
    assert len(labels)==len(points) 

    #Perform voxel-downsampling to reduce all points
    point_cloud=open3d.geometry.PointCloud()
    point_cloud.points=open3d.utility.Vector3dVector(points[:,0:3].copy())
    _,_,indices_list=point_cloud.voxel_down_sample_and_trace(VOXEL_SIZE,point_cloud.get_min_bound(), point_cloud.get_max_bound()) 
    print(f'Reduced to {len(indices_list)} points after voxel-down')
    
    #Not vectorized but seems fast enough, CARE: first-index color sampling (not averaging)
    indices=np.array([ vec[0] for vec in indices_list ])

    #Reduce points by simple removal
    if len(indices)>max_points:
        step=int(np.ceil(len(indices)/max_points))
        assert step>1
        indices=indices[::step]
        print(f'Reduces to {len(indices)} points by simple removal')
    else:
        print('No removal necessary')

    #Save xyz, rgb and labels in separate numpy files
    np.save(open(filepath_xyz,'wb'), np.float32(points[indices,0:3]))
    np.save(open(filepath_rgb,'wb'), np.uint8(  points[indices,4:7]))
    np.save(open(filepath_labels,'wb'), np.uint8(  labels[indices]))

#TODO: handle big scene differently, try read->numpy only or read->reduce->numpy or separate reads
if __name__ == "__main__":
    # scene_name='sg27_station2_intensity_rgb'

    # sg0=np.load(open('sg_s2_0.npy','rb'))
    # sg1=np.load(open('sg_s2_1.npy','rb'))
    # sg2=np.load(open('sg_s2_2.npy','rb'))
    # sg3=np.load(open('sg_s2_3.npy','rb'))
    # sg4=np.load(open('sg_s2_4.npy','rb'))

    # points=np.vstack(( sg0,sg1,sg2,sg3,sg4 ))
    # print(len(points),'points')
    # np.save(open('sg_s2_all.npy','wb'), np.float32(points))
    # print('saved')

    # quit()

    #for scene_name in ALL_SCENE_NAMES:
    for scene_name in ('sg27_station2_intensity_rgb',):
        convert_downsample(os.path.join('data','raw',scene_name+'.txt'),
                           os.path.join('data','labels',scene_name+'.labels'),
                           os.path.join('data','numpy',scene_name+'.xyz.npy'),
                           os.path.join('data','numpy',scene_name+'.rgb.npy'),
                           os.path.join('data','numpy',scene_name+'.lbl.npy'))


