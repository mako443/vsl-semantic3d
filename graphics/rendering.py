import numpy as np
import os
import pickle
import open3d
import cv2
import time
import pptk

from main import load_files
from semantic.imports import ClusteredObject
from .imports import Pose, CLASSES_COLORS, CLASSES_DICT, COMBINED_SCENE_NAMES

def reduce_points(points, max_points):
    step=int(np.ceil(len(points)/max_points))
    points=points[::step].copy()
    return points

#Expand labelled points into unknown points: Find the nearest neighbors via KD-Tree, label all currently unkown points like their closest known neighbor
#Better than other logic from main.load_files()?
def expand_labels(xyz, rgb, lbl, iterations):
    pass

#Deprecated? / copy from main, no alpha, no labels
#Load as rgb, treat label and rgb the same
def load_files2(base_path, scene_name, max_points=int(28e6), target='rgb'):
    assert target in ('rgb','labels','depth')
    p_xyz   =os.path.join(base_path,scene_name+'.xyz.npy')
    p_rgb   =os.path.join(base_path,scene_name+'.rgb.npy')
    p_labels=os.path.join(base_path,scene_name+'.lbl.npy')
    print(p_xyz)

    assert os.path.isfile(p_xyz) and os.path.isfile(p_rgb) and os.path.isfile(p_labels)

    #Load numpy files
    xyz, rgb, lbl=np.load(open(p_xyz,'rb')),np.load(open(p_rgb,'rb')),np.load(open(p_labels,'rb'))

    #Iteratively expand the artifacts into unknowns
    k=5
    iterations=2
    print(f"artefacts before: {np.sum(lbl==CLASSES_DICT['scanning artefacts'])/len(xyz):0.3f}")
    kd_tree=pptk.kdtree._build(xyz)
    for i in range(iterations):
        # print('query tree...')
        neighbors=pptk.kdtree._query(kd_tree, lbl==CLASSES_DICT['scanning artefacts'], k=5) #Neighbors for every artifact point, kd-query returns absolute indices apparently
        neighbors=np.array(neighbors).flatten() #All neighbors of artifact points
        if len(neighbors)==0:
            break

        neighbors=neighbors[lbl[neighbors]==CLASSES_DICT['unlabeled']] #Neighbors of artefacts that are unknown
        lbl[neighbors]=CLASSES_DICT['scanning artefacts']
        neighbors=None
    print(f"artefacts after: {np.sum(lbl==CLASSES_DICT['scanning artefacts'])/len(xyz):0.3f}")    

    #TODO: blunt artifact removal ok?
    #Remove artifacts
    mask= lbl!=CLASSES_DICT['scanning artefacts']
    print(f'Retaining {np.sum(mask) / len(xyz) : 0.3} of points after artifact removal, {len(xyz)} total points')
    xyz, rgb, lbl=xyz[mask,:], rgb[mask,:], lbl[mask]

    if target=='labels':
        #Remove all non-labelled
        mask= lbl!=CLASSES_DICT['unlabeled']
        xyz, rgb, lbl=xyz[mask,:], rgb[mask,:], lbl[mask]

        #Color according to labels
        for class_name in CLASSES_DICT.keys():
            rgb[ lbl==CLASSES_DICT[class_name],:]=CLASSES_COLORS[class_name]

    #Reduce the points via stepping to prevent memory erros
    xyz, rgb, lbl=reduce_points(xyz, max_points=max_points), reduce_points(rgb, max_points=max_points), reduce_points(lbl, max_points=max_points)

    return xyz, rgb, lbl

'''
Rendering via 3D clusters
-Unclear if O3D supports RGBA, for label-images w/ O3D possibly split I/O
'''
# def capture_view(visualizer, pose, scene_objects):
#     set_pose(view_control,pose)

def capture_scene(dirpath,split, scene_name, target='rgb'):
    assert target in ('rgb','labels','depth')
    filepath_poses=os.path.join(dirpath,split,scene_name,'poses.pkl')
    #dirpath_out=os.path.join(dirpath,split,scene_name,'rgb')
    dirpath_out=os.path.join(dirpath,split,scene_name,target)
    assert os.path.isfile(filepath_poses) and os.path.isdir(dirpath_out)

    scene_poses=pickle.load( open( filepath_poses, 'rb') )
    poses_rendered={} #Dictionary for the rendered poses, indexed by file name, I,E added

    print(f'Capturing {len(scene_poses)} poses for <{split}> <{scene_name}> <{target}>')
    xyz, rgba, labels_rgba=load_files2('data/numpy_merged/',scene_name, max_points=int(30e6), target=target) #TODO: use load_files from here / new artifact removal, more points? (Seems to help!!)
    labels_rgba=None
    rgb=rgba[:,0:3].copy()
    rgba=None

    point_cloud=open3d.geometry.PointCloud()
    point_cloud.points=open3d.utility.Vector3dVector(xyz)
    point_cloud.colors=open3d.utility.Vector3dVector(rgb/255.0)

    vis = open3d.visualization.Visualizer()
    vis.create_window(width=1620, height=1080)

    view_control=vis.get_view_control()
    vis.get_render_option().background_color = np.asarray([0, 0, 0])    

    vis.add_geometry(point_cloud)
    for i_pose,pose in enumerate(scene_poses):
        print(f'\r Pose {i_pose} of {len(scene_poses)}',end='')    
        I,E=set_pose(view_control,pose)
        update(vis,point_cloud)
        time.sleep(0.5)

        file_name=f'{i_pose:03d}.png'
        if target in ('rgb','labels'):
            vis.capture_screen_image(os.path.join(dirpath_out,file_name),do_render=True)
        else: #As saving the depth as raw arrays leads to too big files, we compress it to uint8-pngs (scenes are smaller than 255 meters)
            depth= np.asarray(vis.capture_depth_float_buffer(do_render=True))
            depth= np.round(depth)
            #print('depth o-o-r:',np.sum(depth>255))
            depth[depth>255]=255
            depth= np.uint8(depth)
            cv2.imwrite( os.path.join(dirpath_out,file_name), depth)

        #Add I,E to the pose, add to dictionary
        pose.I,pose.E=I,E
        poses_rendered[file_name]=pose

    print()
    print('Saving rendered poses...')
    pickle.dump(poses_rendered, open(os.path.join(dirpath,split,scene_name,'poses_rendered.pkl'), 'wb'))
    print('Scene finished!')
    vis.close()


def update(vis,geometry):
    vis.update_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()

def set_pose(view_control, pose : Pose):
    R=np.vstack((-1.0*pose.right, -1.0*pose.up, -1.0*pose.forward))
    t=-1.0*pose.eye
    
    Rt= R@t
    E=np.zeros((4,4))
    E[0:3,0:3]= R
    E[0:3,  3]=Rt
    E[3,3]=1

    params_new=open3d.camera.PinholeCameraParameters(view_control.convert_to_pinhole_camera_parameters())
    params_new.extrinsic=E
    #print(params_new.extrinsic)

    view_control.convert_from_pinhole_camera_parameters(params_new) 
    I,E=params_new.intrinsic.intrinsic_matrix, params_new.extrinsic
    return I,E

# def calculate_scene_size(xyz):
#     num_points=len(xyz)
#     reject_fraction=0.001
#     reject_points=int(reject_fraction*num_points)

#     x_lower_bound=np.min( np.flip(np.sort(xyz[:,0]))[0:num_points-reject_points] )
#     x_upper_bound=np.max( np.sort(xyz[:,0])         [0:num_points-reject_points] )
#     y_lower_bound=np.min( np.flip(np.sort(xyz[:,1]))[0:num_points-reject_points] )
#     y_upper_bound=np.max( np.sort(xyz[:,1])         [0:num_points-reject_points] )
#     z_lower_bound=np.min( np.flip(np.sort(xyz[:,2]))[0:num_points-reject_points] )
#     z_upper_bound=np.max( np.sort(xyz[:,2])         [0:num_points-reject_points] )     
#     return x_upper_bound-x_lower_bound, y_upper_bound-y_lower_bound, z_upper_bound-z_lower_bound
    


if __name__ == "__main__":
    # for scene_name in COMBINED_SCENE_NAMES:
    #     xyz,_,_=load_files('data/numpy_merged/'+scene_name, max_points=int(20e6), remove_artifacts=False, remove_unlabeled=False, expand_artifacts=False)
    #     print(scene_name,calculate_scene_size(xyz))

    # quit()

    poses_train=pickle.load(open('/home/imanox/Documents/Text2Image/Semantic3D-Net/data/pointcloud_images_o3d_merged/train/sg27_station1_intensity_rgb/poses.pkl' ,'rb'))
    poses_test =pickle.load(open('/home/imanox/Documents/Text2Image/Semantic3D-Net/data/pointcloud_images_o3d_merged/test/sg27_station1_intensity_rgb/poses.pkl' ,'rb'))
    quit()

    '''
    Data creation: Open3D rendering for clusters, read poses -> render images
    '''
    #for scene_name in ('domfountain_station1_xyz_intensity_rgb','sg27_station2_intensity_rgb','untermaederbrunnen_station1_xyz_intensity_rgb','neugasse_station1_xyz_intensity_rgb'):
    #for scene_name in enumerate(COMBINED_SCENE_NAMES):
    for split in ('train','test',):
        for scene_name in COMBINED_SCENE_NAMES:
        #for scene_name in ('bildstein_station1_xyz_intensity_rgb',):
            print(f'\n\n Rendering Scene {scene_name} split {split}')
            capture_scene('data/pointcloud_images_o3d_merged_occ/',split,scene_name,target='rgb')
            capture_scene('data/pointcloud_images_o3d_merged_occ/',split,scene_name,target='labels')
            capture_scene('data/pointcloud_images_o3d_merged_occ/',split,scene_name,target='depth')
    quit()

    ### Project & bbox verify 
    # scene_name='domfountain_station1_xyz_intensity_rgb'
    # file_name='016.png'
    # scene_objects=pickle.load( open('data/numpy/'+scene_name+'.objects.pkl', 'rb'))
    # poses_rendered=pickle.load( open( os.path.join('data','pointcloud_images_o3d',scene_name,'poses_rendered.pkl'), 'rb'))
    # img=cv2.imread( os.path.join('data','pointcloud_images_o3d',scene_name,'rgb', file_name) )

    # pose=poses_rendered[file_name]
    # I,E=pose.I, pose.E

    # for obj in scene_objects:
    #     obj.project(pose.I, pose.E)
    #     if obj.mindist_i>0:
    #         obj.draw_on_image(img)

    # cv2.imshow("",img)
    # cv2.waitKey()
    # # quit()

    # xyz, rgba, labels_rgba=load_files('data/numpy/'+scene_name, remove_artifacts=True, remove_unlabeled=True, max_points=int(10e6))
    # point_cloud=open3d.geometry.PointCloud()
    # point_cloud.points=open3d.utility.Vector3dVector(xyz)
    # point_cloud.colors=open3d.utility.Vector3dVector(rgba[:,0:3]/255.0)

    # for obj in scene_objects:
    #     pcd=open3d.geometry.PointCloud()
    #     pcd.points=open3d.utility.Vector3dVector(obj.points_w)
    #     pcd.colors=open3d.utility.Vector3dVector(np.ones_like(obj.points_w)*(1,0,0))

    # vis = open3d.visualization.Visualizer()
    # vis.create_window(width=1620, height=1080)

    # view_control=vis.get_view_control()
    # vis.get_render_option().background_color = np.asarray([0, 0, 0])

    # vis.add_geometry(point_cloud)
    # for obj in scene_objects:
    #     pcd=open3d.geometry.PointCloud()
    #     pcd.points=open3d.utility.Vector3dVector(obj.points_w)
    #     color=np.random.rand(3)
    #     pcd.colors=open3d.utility.Vector3dVector(np.ones_like(obj.points_w)*color)
    #     vis.add_geometry(pcd)    

    # params_new=open3d.camera.PinholeCameraParameters(view_control.convert_to_pinhole_camera_parameters())
    # params_new.extrinsic=pose.E
    # view_control.convert_from_pinhole_camera_parameters(params_new)
    # vis.run()

    # quit()
    ### Project & bbox verify


    scene_name='domfountain_station1_xyz_intensity_rgb'
    scene_objects=pickle.load( open('data/numpy/'+scene_name+'.objects.pkl','rb'))
    scene_poses=pickle.load( open( os.path.join('data','pointcloud_images_o3d',scene_name,'poses.pkl'), 'rb') )

    xyz, rgba, labels_rgba=load_files('data/numpy/'+scene_name, remove_artifacts=True, remove_unlabeled=True, max_points=int(10e6))

    point_cloud=open3d.geometry.PointCloud()
    point_cloud.points=open3d.utility.Vector3dVector(xyz)
    point_cloud.colors=open3d.utility.Vector3dVector(rgba[:,0:3]/255.0)

    vis = open3d.visualization.Visualizer()
    vis.create_window(width=1620, height=1080)

    view_control=vis.get_view_control()
    vis.get_render_option().background_color = np.asarray([0, 0, 0])

    vis.add_geometry(point_cloud)
    vis.run()

    cars=[ o for o in scene_objects if o.label=='cars']
    print('cars',len(cars))

    params=view_control.convert_to_pinhole_camera_parameters()

    img=np.asarray(vis.capture_screen_float_buffer())
    for car in cars:
        car.project(params)
        car.draw_on_image(img)

    cv2.imshow("",img)
    cv2.waitKey()




#RENDERING OPTIONS: 
#Open3D SLURM: Can get Pinhole params, but blacked out or otherwise unfeasible?, Camera setting ok now
#PyVista SLURM: Perspective anders (korrekt?), camera also seems accessible: https://github.com/pyvista/pyvista-support/issues/85, not blacked
#-> Beide waren schlecht zum "Umschauen", Perspektiven leicht verschieden...




'''
Interactive PyVista on SLURM:

srun xvfb-run python3 -i

import numpy as np
import pyvista      

points=np.random.rand(100,3)
mesh=pyvista.PolyData(points)

plotter=pyvista.Plotter(off_screen=True)
_=plotter.add_mesh(mesh)
res=plotter.show(screenshot='pyvista.png', interactive=False)
'''

