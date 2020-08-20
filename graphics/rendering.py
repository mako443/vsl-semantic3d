import numpy as np
import os
import pickle
import open3d
import cv2
import time

from main import load_files
from semantic.imports import ClusteredObject2
from .imports import Pose


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
def load_files2(base_path, scene_name, max_points=int(28e6)):
    p_xyz   =os.path.join(base_path,scene_name+'.xyz.npy')
    p_rgb   =os.path.join(base_path,scene_name+'.rgb.npy')
    p_labels=os.path.join(base_path,scene_name+'.labels.npy')

    assert os.path.isfile(p_xyz) and os.path.isfile(p_rgb) and os.path.isfile(p_labels)

    #Load numpy files
    xyz, rgb, lbl=np.load(open(p_xyz,'rb')),np.load(open(p_rgb,'rb')),np.load(open(p_labels,'rb'))

    #TODO: blunt artifact removal ok?
    #Remove artifacts
    mask= lbl!=CLASSES_DICT['scanning artefacts']
    print(f'Retaining {np.sum(mask) / len(xyz) : 0.3} of points after artifact removal')
    xyz, rgb, lbl=xyz[mask,:], rgb[mask,:], lbl[mask]

    #Reduce the points via stepping to prevent memory erros
    xyz, rgb, lbl=reduce_points(xyz, max_points=max_points), reduce_points(rgb, max_points=max_points), reduce_points(lbl, max_points=max_points)

    return xyz, rgb, lbl

'''
Rendering via 3D clusters
-Unclear if O3D supports RGBA, for label-images w/ O3D possibly split I/O
'''
def capture_view(visualizer, pose, scene_objects):
    set_pose(view_control,pose)

def capture_scene(dirpath, scene_name):
    filepath_poses=os.path.join(dirpath,scene_name,'poses.pkl')
    dirpath_out=os.path.join(dirpath,scene_name,'rgb')
    assert os.path.isfile(filepath_poses) and os.path.isdir(dirpath_out)

    scene_poses=pickle.load( open( filepath_poses, 'rb') )

    print(f'Capturing {len(scene_poses)} poses')
    xyz, rgba, labels_rgba=load_files('data/numpy/'+scene_name, remove_artifacts=True, remove_unlabeled=True, max_points=int(30e6))
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
        set_pose(view_control,pose)
        update(vis,point_cloud)
        time.sleep(0.5)

        img=np.asarray(vis.capture_screen_float_buffer())
        img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        file_name=f'{i_pose:03d}.png'
        cv2.imwrite( os.path.join(dirpath_out,file_name), img*255)
    print()
    print('Scene finished!')


def update(vis,geometry):
    vis.update_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()

def set_pose(view_control, pose : Pose):
    # view=np.array([ 0.57511449, -0.81660008,  0.04906747])
    # up=np.array([-0.02825345,  0.04011682,  0.99879545])
    # right=np.array([0.81758493, 0.57580805, 0.        ])   
    # eye=np.array([-1.32071638,  0.7591362 ,  0.24045286])

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


if __name__ == "__main__":
    '''
    Open3D rendring for clusters: read poses -> render images
    '''
    # scene_name='domfountain_station1_xyz_intensity_rgb'
    # capture_scene('data/pointcloud_images_o3d/',scene_name)


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
#Open3D SLURM: Can get Pinhole params, but blacked out or otherwise unfeasible?
#PyVista SLURM: Perspective anders (korrekt?), camera also seems accessible: https://github.com/pyvista/pyvista-support/issues/85, not blacked
#-> Beide waren schlecht zum "Umschauen", Perspektiven leicht verschieden...

'''
TODO
-Possibly try SLURM again? -> No ✖, maybe if voxel-downsample fails
-build from main&capturing, remove capturing
-voxel-downsample before rendering? Then on PyVista or Open3D?
'''



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



