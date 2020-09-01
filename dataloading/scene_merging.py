import numpy as np
import open3d
import os
import pptk
import sys
import pickle
from scipy.spatial.transform import Rotation
from graphics.rendering import load_files2
from graphics.imports import CLASSES_DICT
from .data_preparation import VOXEL_SIZE

'''
Module to merge point clouds using Open3D, source: http://www.open3d.org/docs/release/tutorial/Advanced/multiway_registration.html
'''

#List of dictionaries {scene_name: (angle, shift)}, angle as degrees around z-axis, shift as [x,y,z] #TODO: don't need x/y rotation?
COMBINED_SCENES=[]
COMBINED_SCENES.append({
    'bildstein_station1_xyz_intensity_rgb': (0, [0,0,0]), 
    'bildstein_station3_xyz_intensity_rgb': (-172,[22,-32,3]),
    'bildstein_station5_xyz_intensity_rgb': (3,[60,28,0]),
})

def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = open3d.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        open3d.registration.TransformationEstimationPointToPlane())
    icp_fine = open3d.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        open3d.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = open3d.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = open3d.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(open3d.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build open3d.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    open3d.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    open3d.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    open3d.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=True))
    return pose_graph

def view_together(xyz0, rgb0, xyz1, rgb1, z_degrees, shift):
    R=Rotation.from_euler('z',z_degrees,degrees=True).as_matrix()
    xyz1_r=xyz1.copy()

    xyz1_r=xyz1_r@R
    xyz1_r+=shift

    xyz=np.vstack(( xyz0,xyz1_r ))
    rgb=np.vstack(( rgb0, rgb1 ))

    v=pptk.viewer(xyz)
    v.attributes(rgb/255.0)
    v.set(point_size=0.04)

#CARE: be carefull to check that the registration converged!
VOXEL_SIZE_MERGE=5*VOXEL_SIZE #Further downsample for normal-estimation and pose-graph creation, actual merging and new down-sampling is done at VOXEL_SIZE
max_correspondence_distance_coarse = VOXEL_SIZE_MERGE * 15 * 3
max_correspondence_distance_fine = VOXEL_SIZE_MERGE * 1.5 * 3

def find_pose_graph(combine_dict):
    #Find the combined pose-graph for all scenes
    scene_names=sorted(list(combine_dict.keys()))
    print('Finding pose-graph for scenes',scene_names)
    point_clouds=[]
    point_clouds_full=[]

    #Load all scenes, pre shift&rotate, downsample even more for registration, compute normals
    for scene_name in scene_names:
        z_angle,shift=combine_dict[scene_name]
        xyz, rgb, lbl=load_files2('data/numpy/',scene_name)
        R=Rotation.from_euler('z',z_angle,degrees=True).as_matrix()
        xyz=xyz@R
        xyz+=shift

        pcd=open3d.geometry.PointCloud()
        pcd.points=open3d.utility.Vector3dVector(xyz)
        pcd.colors=open3d.utility.Vector3dVector(rgb/255.0)
        print(pcd)
        point_clouds_full.append(pcd)

        pcd_down=pcd.voxel_down_sample(VOXEL_SIZE_MERGE)
        pcd_down.estimate_normals()

        point_clouds.append(pcd_down)  

    #open3d.visualization.draw_geometries(point_clouds)

    #Do the registration
    print("Full registration ...")
    with open3d.utility.VerbosityContextManager(open3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(point_clouds,
                                    max_correspondence_distance_coarse,
                                    max_correspondence_distance_fine)

    print("Optimizing PoseGraph ...")
    option = open3d.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    with open3d.utility.VerbosityContextManager(open3d.utility.VerbosityLevel.Debug) as cm:
        open3d.registration.global_optimization(
            pose_graph, open3d.registration.GlobalOptimizationLevenbergMarquardt(),
            open3d.registration.GlobalOptimizationConvergenceCriteria(), option)

    for i,point_cloud in enumerate(point_clouds):
        point_cloud.transform(pose_graph.nodes[i].pose)

    for i,point_cloud in enumerate(point_clouds_full):
        print(point_cloud)
        point_cloud.transform(pose_graph.nodes[i].pose)
        xyz=np.asarray(point_cloud.points)
        np.save(open(scene_names[i]+'.shifted.npy', 'wb'), xyz)

    open3d.visualization.draw_geometries(point_clouds)  

    return pose_graph  
    
def apply_transform(xyz,pose):
    pcd=open3d.geometry.PointCloud()
    pcd.points=open3d.utility.Vector3dVector(xyz)
    pcd.transform(pose)
    xyz=np.asarray(pcd.points)
    return xyz

#Load all scenes again, combine, Voxel-down again in combination
def combine_scene(combine_dict, pose_graph):
    scene_names=sorted(list(combine_dict.keys()))
    print('Combining scenes',scene_names)

    xyz_combined=np.array([],dtype=np.float32).reshape((0,3))
    rgb_combined=np.array([],dtype=np.uint8).reshape((0,3))
    lbl_combined=np.array([],dtype=np.uint8)

    for i,scene_name in enumerate(scene_names):
        z_angle,shift=combine_dict[scene_name]
        xyz, rgb, lbl=load_files2('data/numpy/',scene_name)
        
        #Apply the transformation from the manual alignment
        R=Rotation.from_euler('z',z_angle,degrees=True).as_matrix()
        xyz=xyz@R
        xyz+=shift    

        #Apply the pose-graph transformation
        #xyz=apply_transform(xyz,poses[i])
        xyz=apply_transform(xyz,pose_graph.nodes[i].pose)

        xyz_combined=np.vstack(( xyz_combined, xyz ))
        rgb_combined=np.vstack(( rgb_combined, rgb ))
        lbl_combined=np.hstack(( lbl_combined, lbl ))

    assert len(xyz_combined) == len(rgb_combined) == len(lbl_combined)
    #xyz_combined, rgb_combined, lbl_combined=xyz_combined[::2].copy(), rgb_combined[::2].copy(), lbl_combined[::2].copy()

    print(f'{len(xyz_combined)} points combined, voxel-down again')
    point_cloud=open3d.geometry.PointCloud()
    point_cloud.points=open3d.utility.Vector3dVector(xyz_combined)
    
    _,_,indices_list=point_cloud.voxel_down_sample_and_trace(VOXEL_SIZE,point_cloud.get_min_bound(), point_cloud.get_max_bound()) 
    print(f'Reduced to {len(indices_list)} points after voxel-down')
    indices=np.array([ vec[0] for vec in indices_list ])

    return xyz_combined[indices,:], rgb_combined[indices,:], lbl_combined[indices]


'''
TODO
-out of the box: not working ✖
-just pairwise: not working ✖
-same scene w/ shift: small shift works ✓ big shift doesn't work ✓ 90° rot doesn't work ✓ 5° rot works ✓
-pre rot&shift, all np math: seems to work, hope this is reliable ¯\_(ツ)_/¯

-merge complete scene -> full pipeline -> check renderings & sgs
'''

if __name__ == "__main__":
    '''
    Data creation: Pre-align scenes manually
    '''
    if sys.argv[1]=='align':
        base_path='data/numpy/'
        source_scene_name=sys.argv[2]
        target_scene_name=sys.argv[3]

        print(f'Manual alignment for {source_scene_name} and {target_scene_name}')

        xyz0, rgb0, lbl0=load_files2('data/numpy',source_scene_name)
        xyz0,rgb0=xyz0[lbl0==CLASSES_DICT['buildings']],rgb0[lbl0==CLASSES_DICT['buildings']]
        rgb0[:,0]=255      

        xyz1, rgb1, lbl1=load_files2('data/numpy',target_scene_name)
        xyz1,rgb1=xyz1[lbl1==CLASSES_DICT['buildings']],rgb1[lbl1==CLASSES_DICT['buildings']]
        rgb1[:,2]=255                

        view_together(xyz0, rgb0, xyz1, rgb1, 0, [0,0,0])

    '''
    Data creation: Read, align and combine scenes
    '''
    if sys.argv[1]=='combine':
        base_in ='data/numpy/'
        base_out='data/numpy_combined/'

        for combine_dict in COMBINED_SCENES:
            pose_graph=find_pose_graph(combine_dict)
            xyz,rgb,lbl=combine_scene(combine_dict, pose_graph)
            save_name=sorted(list(combine_dict.keys()))[0]
            print()
            print('Saving combined scene',save_name)
            np.save(open(os.path.join(base_out,save_name+'.xyz.npy'),'wb'), np.float32(xyz))
            np.save(open(os.path.join(base_out,save_name+'.rgb.npy'),'wb'), np.uint8(rgb))
            np.save(open(os.path.join(base_out,save_name+'.lbl.npy'),'wb'), np.uint8(lbl))

    #TODO: iterate through scenes, just copy if only one sub-scene

    # xyz, rgb, lbl=load_files2('data/numpy/','bildstein_station3_xyz_intensity_rgb')
    # source=open3d.geometry.PointCloud()
    # source.points=open3d.utility.Vector3dVector(xyz)
    # source.colors=open3d.utility.Vector3dVector(rgb/255.0)
    # source_down=source.voxel_down_sample(VOXEL_SIZE_MERGE)
    # source_down.estimate_normals()

    # R=Rotation.from_euler('z',-195,degrees=True).as_matrix()
    # shift=[-25,-67.5,-4]
    # xyz, rgb, lbl=load_files2('data/numpy/','bildstein_station5_xyz_intensity_rgb')
    # xyz=xyz@R
    # xyz+=shift

    # target=open3d.geometry.PointCloud()
    # target.points=open3d.utility.Vector3dVector(xyz)
    # target.colors=open3d.utility.Vector3dVector(rgb/255.0)
    # target_down=target.voxel_down_sample(VOXEL_SIZE_MERGE)
    # target_down.estimate_normals()    

    # point_clouds=[source_down,target_down]
    
    # open3d.visualization.draw_geometries(point_clouds)
    # # quit()

    # #Target: Residual near 0, fitness near 1.0

    # max_correspondence_distance_coarse = VOXEL_SIZE_MERGE * 15 #CARE: This correspondence distance the problem?
    # max_correspondence_distance_fine = VOXEL_SIZE_MERGE * 1.5
    # max_correspondence_distance_coarse *=3
    # max_correspondence_distance_fine *=3

    # print("Full registration ...")
    # with open3d.utility.VerbosityContextManager(open3d.utility.VerbosityLevel.Debug) as cm:
    #     pose_graph = full_registration(point_clouds,
    #                                 max_correspondence_distance_coarse,
    #                                 max_correspondence_distance_fine)

    # print("Optimizing PoseGraph ...")
    # option = open3d.registration.GlobalOptimizationOption(
    #     max_correspondence_distance=max_correspondence_distance_fine,
    #     edge_prune_threshold=0.25,
    #     reference_node=0)
    # with open3d.utility.VerbosityContextManager(open3d.utility.VerbosityLevel.Debug) as cm:
    #     open3d.registration.global_optimization(
    #         pose_graph, open3d.registration.GlobalOptimizationLevenbergMarquardt(),
    #         open3d.registration.GlobalOptimizationConvergenceCriteria(), option)

    # for i,point_cloud in enumerate(point_clouds):
    #     print()
    #     print('before',point_cloud.get_min_bound())
    #     point_cloud.transform(pose_graph.nodes[i].pose)            
    #     print('after ',point_cloud.get_min_bound())

    # open3d.visualization.draw_geometries(point_clouds)





    

    
