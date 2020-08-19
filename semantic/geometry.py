import numpy as np
import os
#from graphics.rendering import Pose

#CARE: Only define here, import everywhere else
#FOV estimated "by hand"
FOV_W=64.0
FOV_H=44.8
IMAGE_WIDHT=1620
IMAGE_HEIGHT=1080

#CARE: projection with E has some inaccuracy and needs coordinate-system correction...
#I from: https://codeyarns.com/2015/09/08/how-to-compute-intrinsic-camera-matrix-for-a-camera/
#E from: https://math.stackexchange.com/questions/2062633/how-can-i-find-extrinsic-camera-matrix
def get_camera_matrices(pose):
    #Intrinsic matrix
    I=np.zeros((3,3))
    I[0,0]=IMAGE_WIDHT/2.0 / np.tan( np.deg2rad(FOV_W/2.0) ) #f_x
    I[1,1]=IMAGE_HEIGHT/2.0 / np.tan( np.deg2rad(FOV_H/2.0) ) #f_y
    I[0,1]=0.0 #s
    I[0,2]=IMAGE_WIDHT/2.0 #x
    I[1,2]=IMAGE_HEIGHT/2.0 #y
    I[2,2]=1.0

    #Extrinsic matrix
    #view,up,right=viewer.get('view'), viewer.get('up'), viewer.get('right')
    view,up,right=pose.forward, pose.up, pose.right
    R=np.vstack((right, up, view))
    t=-pose.eye
    Rt= np.reshape(R@t,(3,1))
    E=np.hstack((R,Rt))

    return I,E

def project_point(I,E,point):
    point=I@E@np.hstack((point,1))
    return np.array(( IMAGE_WIDHT-point[0]/point[2], point[1]/point[2], -point[2] ))    

def is_object_in_fov(obj):
    assert obj.bbox_points_i is not None
    return obj.maxdist_i>0 and (obj.bbox_points_i[:,0].min()>0.1*IMAGE_WIDHT or obj.bbox_points_i[:,0].max()<0.9*IMAGE_WIDHT) and (obj.bbox_points_i[:,1].min()>0.1*IMAGE_HEIGHT or obj.bbox_points_i[:,1].max()<0.9*IMAGE_WIDHT)
    
    