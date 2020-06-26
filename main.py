import numpy as np
import os
import pptk
import time
import sys

import capturing

# convert_txt('data/xyz_rgb/bildstein_station3_xyz_intensity_rgb.txt',
#             'data/labels/bildstein_station3_xyz_intensity_rgb.labels',
#             'data/numpy/bildstein_station3_xyz_intensity_rgb.xyz.npy',
#             'data/numpy/bildstein_station3_xyz_intensity_rgb.rgb.npy',
#             'data/numpy/bildstein_station3_xyz_intensity_rgb.labels.npy')

classes_dict={'unlabeled': 0, 'man-made terrain': 1, 'natural terrain': 2, 'high vegetation': 3, 'low vegetation': 4, 'buildings': 5, 'hard scape': 6, 'scanning artefacts': 7, 'cars': 8}
classes_colors={'unlabeled': (255,255,255), 'man-made terrain': (50,30,30), 'natural terrain': (30,50,30), 'high vegetation': (120,255,120), 'low vegetation': (80,255,80), 'buildings': (255,255,0), 'hard scape': (0,255,255), 'scanning artefacts': (255,0,0), 'cars': (0,0,255)}

def convert_txt(filepath_points, filepath_labels_in, filepath_xyz, filepath_rgb, filepath_labels_out):
    assert os.path.isfile(filepath_points)
    assert os.path.isfile(filepath_labels_in)
    assert not os.path.isfile(filepath_xyz)
    assert not os.path.isfile(filepath_rgb)
    assert not os.path.isfile(filepath_labels_out)

    print('\nloading points...')
    point_cloud=np.loadtxt(filepath_points, delimiter=' ', dtype=np.float32)
    num_points=point_cloud.shape[0]
    print(f'loading points done, {num_points} points')

    xyz=point_cloud[:,0:3]
    np.save(open(filepath_xyz,'wb'), xyz)
    xyz=None
    print('xyz saved.')

    rgb=point_cloud[:,4:].astype(np.uint8)
    np.save(open(filepath_rgb,'wb'), rgb)
    rgb=None
    print('rgb saved.')

    print('loading labels...')
    labels=np.loadtxt(filepath_labels_in, dtype=np.uint8).flatten()
    print('num labels:', labels.shape)
    np.save(open(filepath_labels_out, 'wb'), labels)
    labels=None
    print('labels saved')

    point_cloud=None

def load_files(base_path, halve_points=False, remove_artifacts=True, remove_unlabeled=True, expand_artifacts=True):
    p_xyz=base_path+'.xyz.npy'
    p_rgb=base_path+'.rgb.npy'
    p_labels=base_path+'.labels.npy'

    assert os.path.isfile(p_xyz) and os.path.isfile(p_rgb) and os.path.isfile(p_labels)

    xyz, rgb, lbl=np.load(open(p_xyz,'rb')),np.load(open(p_rgb,'rb')),np.load(open(p_labels,'rb'))
    rgba= np.hstack((rgb, 255*np.ones((len(rgb),1))))
    rgb=None #Clear memory

    if halve_points:
        print('halving points')
        xyz, rgba, lbl= xyz[::2], rgba[::2], lbl[::2]
    
    #Iteratively expand the artifacts into unknowns
    k=5
    iterations=2
    if expand_artifacts:
        print(f"artefacts before: {np.sum(lbl==classes_dict['scanning artefacts'])/len(rgba):0.3f}")
        kd_tree=pptk.kdtree._build(xyz)
        for i in range(iterations):
            print('query tree...')
            neighbors=pptk.kdtree._query(kd_tree, lbl==classes_dict['scanning artefacts'], k=5) #Neighbors for every artifact point, kd-query returns absolute indices apparently
            neighbors=np.array(neighbors).flatten() #All neighbors of artifact points
            neighbors=neighbors[lbl[neighbors]==classes_dict['unlabeled']] #Neighbors of artefacts that are unknown
            lbl[neighbors]=classes_dict['scanning artefacts']
            neighbors=None
        print(f"artefacts after: {np.sum(lbl==classes_dict['scanning artefacts'])/len(rgba):0.3f}")

    labels_rgba=rgba.copy()
    for k in classes_dict.keys():
        mask= lbl==classes_dict[k]
        labels_rgba[mask,0:3]=classes_colors[k]            

    #Hide artifacts in rgba and labels_rgba
    if remove_artifacts:
        mask= lbl==classes_dict['scanning artefacts']
        rgba[mask,3]=0
        labels_rgba[mask,3]=0
        print(f'hidden {np.sum(mask)/len(rgba):0.3f} artifacts (in rgba and labels_rgba)')

    #Hide unlabeled in labels_rgba
    if remove_unlabeled:
        mask = lbl==classes_dict['unlabeled']
        #rgba[mask,3]=0
        labels_rgba[mask,3]=0
        print(f'hidden {np.sum(mask)/len(rgba):0.3f} unlabeled (in labels_rgba)')

    return xyz, rgba, labels_rgba

# def capture_360(viewer, name, point_size_color=0.013, point_size_labels=0.026, num_angles=4):
#     assert viewer.get('num_attributes')[0]==2

#     viewer.set(show_grid=False, show_info=False, show_axis=False)
#     viewer.set(bg_color=(0,0,0,1))
#     viewer.set(bg_color_bottom=(0,0,0,1))
#     viewer.set(bg_color_top=(0,0,0,1))


#     st=2.0
#     #Render color images
#     viewer.set(curr_attribute_id=0)
#     viewer.set(point_size=point_size_color)
#     time.sleep(st)
#     for i,phi in enumerate(np.linspace(np.pi, 0,num_angles+1)[0:-1]): 
#         viewer.set(phi=phi,theta=0.0,r=5.0)
#         time.sleep(st)
#         viewer.capture(f'{name}_{i:02d}_color.png')

#     #Render label images
#     viewer.set(curr_attribute_id=1)
#     viewer.set(point_size=point_size_labels)
#     time.sleep(st)
#     for i,phi in enumerate(np.linspace(np.pi, 0,num_angles+1)[0:-1]): 
#         viewer.set(phi=phi,theta=0.0,r=5.0)
#         time.sleep(st)
#         viewer.capture(f'{name}_{i:02d}_label.png')        

#     viewer.set(show_grid=True, show_info=True, show_axis=True)

def view_pptk(base_path, halve_points=False, remove_artifacts=False, remove_unlabeled=True):
    xyz, rgba, labels_rgba=load_files(base_path,halve_points=halve_points, remove_artifacts=remove_artifacts, remove_unlabeled=remove_unlabeled)

    viewer=pptk.viewer(xyz)
    viewer.attributes(rgba.astype(np.float32)/255.0,labels_rgba.astype(np.float32)/255.0)

    #viewer.color_map('hsv')
    viewer.set(point_size=0.013)
    return viewer

def find_unlabeled_artifacts(xyz,lbl):
    unknown_labels=  lbl==classes_dict['unlabeled']
    artifact_labels= lbl==classes_dict['scanning artefacts']
    
    kd_tree=pptk.kdtree._build(xyz)    
    neighbour_indices=pptk.kdtree._query(kd_tree,unknown_labels,k=5) #query the K-NN indices for each unlabeled point
    neighbour_indices=np.array(neighbour_indices)

    neighbour_labels=np.take(lbl, neighbour_indices) # [N,k] array of the labels of the k neighbours each

scene_name='domfountain_station1_xyz_intensity_rgb'
viewer=view_pptk('data/numpy/'+scene_name,halve_points=True, remove_artifacts=True, remove_unlabeled=True)
base_path='data/pointcloud_images/'+scene_name+'/'

input('Enter to continue...')
points=capturing.scene_config[scene_name]['points']
point_size=capturing.scene_config[scene_name]['point_size_rgb']
poses=capturing.points2poses(points,25)
capturing.capture_poses(viewer,base_path+'rgb',base_path+'lbl',base_path+'poses.npy',poses,point_size,2*point_size,8)


#capturing.capture_poses(viewer,'t','rgb','lbl',poses[0:3],0.025,0.05,4)

'''
TODO:
-what's up with the 2 colors mixed? -> many unlabeled points
-show L&Q -> done, Q approves
-render on SLURM-> not possible ✖
-possible to remove unlabeled near artifacts? -> yes, at least partly by kd-tree ✓ 
-load remaining training scenes -> Done on SLURM

-attempt NetVLAD on 200 pics, w/ and w/o label:
-compare quality full points at dense scene / otherwise:
-(PyntCloud/VTK testen: meshed rendering besser? on slurm? dynamic point size?)
-automatically render images&seg-images from all scenes, start w/ best-case stuff, clear floors&holes w/ color-masks

'''

# base_name=sys.argv[1]
# print('converting',base_name)
# convert_txt(f'data/raw/{base_name}.txt',f'data/labels/{base_name}.labels',f'data/numpy/{base_name}.xyz.npy', f'data/numpy/{base_name}.rgb.npy', f'data/numpy/{base_name}.labels.npy')
# print('done!\n\n')

# print('loading pc...')
# xyz=np.load(open('xyz_full.npy','rb'))
# rgb=np.load(open('rgb_full.npy','rb'))
# lbl=np.load(open('lbl_full.npy','rb'))

# viewer=pptk.viewer(xyz,lbl)
# viewer.attributes(rgb.astype(np.float32)/255.0)



# Sem3d
# -per image render from multiple locations with a full circle each
# -retrieval based on SG&NetVLAD, localization based on NetVLAD
# -possibly guide NetVLAD training w/ high-level semantics

# -NetVLAD failures (vis. distinct objects ranked as similar)

# Aachen
# -check NetVLAD pairs (in dataset or Torstens github, Quinjie link)

# for n in ('tc0_01', 'tc1_00', 'tc2_03'):
#     img=cv2.imread(n+'_color.png')
#     lbl=cv2.imread(n+'_label.png')
#     i=np.hstack((img,lbl))
#     cv2.imwrite(n+'.jpg',i)
