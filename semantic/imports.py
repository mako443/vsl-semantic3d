import numpy as np
import os
import cv2

# def project_point(I,E,point):
#     point=I@E@np.hstack((point,1))
#     return np.array(( IMAGE_WIDHT-point[0]/point[2], point[1]/point[2], -point[2] )) 

#TODO: vectorize
def project_point(I,E,point):
    p= E@np.hstack((point,1))
    p= I@p[0:3]
    return np.array(( p[0]/p[2], p[1]/p[2], p[2]))

class ClusteredObject2:
    def __init__(self, label, bbox_points, num_points):
        self.label=label
        self.bbox_points_w=bbox_points # BBox in world-coords, 8x 3d-points
        self.bbox_points_i=None
        self.bbox_rect_i=None # BBox in image-coords, available after self.project(), cv2.RotatedRect object
        self.mindist_i, self.maxdist_i= None, None #Distance in image-coords, available after self.project
        self.num_points=num_points

    def __str__(self):
        return f'ClusteredObject2: {self.label} at {np.mean(self.bbox_points_w, axis=0)}, {self.num_points} points'

    def project(self, params):
        I,E = params.intrinsic.intrinsic_matrix, params.extrinsic
        #Project all world-coordinate points to 3d image-points
        self.bbox_points_i = np.array( [project_point(I,E, point) for point in self.bbox_points_w] )
        #Set bbox_i as rotated rect in image-plane
        self.bbox_rect_i=cv2.minAreaRect(self.bbox_points_i[:,0:2].astype(np.float32)) #Bug w/o float32
        #Set image-plane distances
        self.mindist_i, self.maxdist_i = np.min(self.bbox_points_i[:,2]), np.max(self.bbox_points_i[:,2])

    # is_in_fov() and is_occluded() both external

    def draw_on_image(self,img):
        assert self.bbox_points_i is not None
        box=np.int0(cv2.boxPoints(self.bbox_rect_i))
        cv2.drawContours(img,[box],0,(255,255,0),thickness=2)
        #_=cv2.putText(img, self.label, (points[0,0]+5, points[0,1]+5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0))