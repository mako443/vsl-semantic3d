import pptk
import numpy as np
import os
import time
import semantic.clustering
from main import classes_colors
import cv2

#CARE: Only define here, import everywhere else
#FOV estimated "by hand"
FOV_W=64.0
FOV_H=44.8
IMAGE_WIDHT=1620
IMAGE_HEIGHT=1080

def viewer_to_image(viewer):
    viewer.capture('tmp.png')
    time.sleep(1)
    return cv2.imread('tmp.png')

def resize_window(width=1620,height=1080):
    os.system(f'wmctrl -r viewer -e 0,100,100,{width},{height}')

def reduce_points(points, max_points):
    step=int(np.ceil(len(points)/max_points))
    points=points[::step,:].copy()
    return points

#FIELDS_OF_VIEW={}
#FIELDS_OF_VIEW[(1620,1080)]=(64.0, 44.8)

#Returns the total fov angle in degrees
def calc_fov(radius, camera_y):
    return 2*np.rad2deg( np.arctan2(radius, camera_y) )

#Assuming a pinhole-model
def calc_intrinsic_camera_matrix():
    mat=np.zeros((3,3))
    mat[0,0]=IMAGE_WIDHT/2.0 / np.tan( np.deg2rad(FOV_W/2.0) ) #f_x
    mat[1,1]=IMAGE_HEIGHT/2.0 / np.tan( np.deg2rad(FOV_H/2.0) ) #f_y
    mat[0,1]=0.0 #s
    mat[0,2]=IMAGE_WIDHT/2.0 #x
    mat[1,2]=IMAGE_HEIGHT/2.0 #y
    mat[2,2]=1.0
    return mat

def calc_extrinsic_camera_matrix(viewer):
    view,up,right=viewer.get('view'), viewer.get('up'), viewer.get('right')

    R=np.vstack((right, up, view))
    t=-viewer.get('eye')
    Rt= np.reshape(R@t,(3,1))
    mat=np.hstack((R,Rt))
    return mat

def get_camera(viewer):
    return calc_intrinsic_camera_matrix(), calc_extrinsic_camera_matrix(viewer)

def project_point(I,E,point):
    point=I@E@np.hstack((point,1))
    return np.array(( IMAGE_WIDHT-point[0]/point[2], point[1]/point[2], -point[2] ))
    # point[0:2]/=point[2]
    # point[0]=IMAGE_WIDHT-point[0]
    # point[2]*=-1
    # return point

#DEPRECATED
def calc_position_in_frame(eye, theta, phi, image_size, object_location):
    fov=FIELDS_OF_VIEW[image_size]

#TODO: convert to rotated rect / only lines / remove
def render_objects(clustered_objects):
    points=[]
    grid_points=5
    for o in clustered_objects:
        bbox=o.bbox
        grid=np.linspace(-0.5,0.5,grid_points)
        x,y,z=np.meshgrid(grid,grid,grid)
        xyz=np.array([x.flatten(),y.flatten(), z.flatten()]).T
        points.extend( np.array((bbox[0],bbox[1],bbox[2])) + xyz )
        points.extend( np.array((bbox[0],bbox[1],bbox[5])) + xyz )
        points.extend( np.array((bbox[0],bbox[4],bbox[2])) + xyz )
        points.extend( np.array((bbox[0],bbox[4],bbox[5])) + xyz )
        points.extend( np.array((bbox[3],bbox[1],bbox[2])) + xyz )
        points.extend( np.array((bbox[3],bbox[1],bbox[5])) + xyz )
        points.extend( np.array((bbox[3],bbox[4],bbox[2])) + xyz )
        points.extend( np.array((bbox[3],bbox[4],bbox[5])) + xyz )
    points=np.array(points)
    
    rgba=np.zeros((points.shape[0],4))
    step=grid_points**3
    for i,o in enumerate(clustered_objects):
        rgba[i*step: (i+1)*step,0:3]=classes_colors[o.label]
    rgba[:,3]=255


    return points,rgba

#NEXT: intrinsic check, then extrinsic all possibilities
#Axis swap back&forth?
#Theta is up/down
if __name__ == "__main__":
    radius=10 #Radius is the distance from the point to the middle, verified same result with 2 different radii
    xyz=np.array([ [-radius,radius,radius], [radius,radius,radius] ])
    # xyz=np.array([ [0,0,radius], [0,0,-radius] ])
    colors=np.array([ [1,0,0], [1,0,0] ])

    viewer=pptk.viewer(xyz, colors)
    viewer.set(point_size=0.2)
    viewer.set(lookat=(0,0,0))

    resize_window()
    time.sleep(1)
    
    #I,E=get_camera(viewer)

    

    