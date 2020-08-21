import numpy as np
import os
import cv2
from graphics.imports import IMAGE_WIDHT, IMAGE_HEIGHT, CLASSES_COLORS

RELATIONSHIP_TYPES=('left','right','below','above','infront','behind')
DEPTH_DIST_FACTOR=IMAGE_WIDHT/255.0*2

#TODO: add dark-red (0.5,0,0), do as 8 corners of unit-cube, score relative to max qube-dist
COLOR_NAMES=('red','green','blue','black','white')
COLORS=np.array(( (1,0,0), (0,1,0), (0,0,1), (0,0,0), (1,1,1) )).reshape((5,3))

CORNER_NAMES=('top-left','top-right','bottom-left','bottom-right','center')
CORNERS=np.array(( (0.2, 0.2), (0.8,0.2), (0.2,0.8), (0.8,0.8), (0.5,0.5) )).reshape((5,2)) #Corners as relative (x,y) positions

# def project_point(I,E,point):
#     point=I@E@np.hstack((point,1))
#     return np.array(( IMAGE_WIDHT-point[0]/point[2], point[1]/point[2], -point[2] )) 

#TODO: vectorize
#CARE: Open3D flips axes in the E matrix when setting the Pose! -> x-flip necessary here
def project_point(I,E,point):
    p= E@np.hstack((point,1))
    p= I@p[0:3]
    return np.array(( IMAGE_WIDHT-p[0]/p[2], p[1]/p[2], p[2]))
    #return np.array(( p[0]/p[2], p[1]/p[2], p[2]))

class ClusteredObject:
    def __init__(self, scene_name, label, points_w, total_points, color):
        self.scene_name=scene_name
        self.label=label
        self.points_w=points_w #3D world-coordinate points, possibly reduced | convex-hull not possible because of projection errors

        # self.points_i=None #3D image-coordinate points, available after self.project() #TODO: remove?
        self.rect_i=None #Rotated bounding-box in image-coords, available after self.project(), cv2.RotatedRect object
        self.mindist_i, self.maxdist_i= None, None #Distance in image-coords, available after self.project()
        
        self.total_points=total_points #Original number of points (as opposed to hull)
        self.color=color #Color as [r,g,b] in [0,1]

    def __str__(self):
        return f'ClusteredObject: {self.label} at {np.mean(self.points_w, axis=0)}, {self.total_points} points'

    def project(self, I, E):
        points_i= np.array( [ project_point(I,E, point) for point in self.points_w ] ) #Causes projection instabilities when points are out of FoV
        mask=np.bitwise_and.reduce(( points_i[:,0]>=0, points_i[:,0]<=IMAGE_WIDHT, points_i[:,1]>=0, points_i[:,1]<=IMAGE_HEIGHT, points_i[:,2]>0  )) #Mask to clamp to visible region
        points_i=points_i[mask, :].copy()
        if len(points_i)>0: #Projection sucessful, partly in fov
            self.rect_i=cv2.minAreaRect(points_i[:,0:2].astype(np.float32))
            self.mindist_i, self.maxdist_i = np.min(points_i[:,2]), np.max(points_i[:,2])
            self.points_i=points_i
        else: #Projection failed, none in FoV
            self.rect_i=None
            self.mindist_i, self.maxdist_i= None, None

    def get_area(self):
        assert self.rect_i is not None
        center, lengths, angle=self.rect_i
        return np.prod(lengths)

    def draw_on_image(self,img):
        assert self.rect_i is not None
        box=np.int0(cv2.boxPoints(self.rect_i))
        cv2.drawContours(img,[box],0,(255,255,0),thickness=2)
        #for p in self.points_i:
        #    _=cv2.circle(img, (int(p[0]), int(p[1])), 6, color=(255,0,255), thickness=3)
        #_=cv2.putText(img, self.label, (points[0,0]+5, points[0,1]+5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0))

class ViewObject:
    #__slots__ = ['label', 'bbox', 'depth', 'center', 'color']
    __slots__ = ['scene_name', 'label', 'points','rect', 'mindist', 'maxdist', 'center', 'color','min_z_w','max_z_w']

    # @classmethod
    # def from_patch(cls, patch):
    #     v=ViewObject()
    #     v.label=patch.label
    #     v.bbox=patch.bbox #CARE: Center is normed, BBbox is not!
    #     v.depth=patch.depth
    #     v.center=patch.center/(IMAGE_WIDHT, IMAGE_HEIGHT) #Convert center [0,IMAGE_W/H] -> [0,1]
    #     v.color=patch.color/255.0 #Convert color [0,255] -> [0,1]
    #     return v
    @classmethod
    def from_clustered_object(cls, clustered_object : ClusteredObject):
        v=ViewObject()
        assert clustered_object.points_i is not None
        assert clustered_object.rect_i is not None
        v.scene_name=clustered_object.scene_name
        v.label=clustered_object.label
        v.points=clustered_object.points_i #Projected Convex-hull points #TODO: remove?
        v.rect=clustered_object.rect_i
        v.mindist=clustered_object.mindist_i
        v.maxdist=clustered_object.maxdist_i
        v.min_z_w, v.max_z_w= np.min(clustered_object.points_w[:,2]), np.max(clustered_object.points_w[:,2])
        v.center=np.array(v.rect[0]) #CARE: Not re-created yet!
        v.color=clustered_object.color #Color as [r,g,b] in [0,1]
        return v

    def draw_on_image(self,img):
        box=np.int0(cv2.boxPoints(self.rect))
        color=CLASSES_COLORS[self.label]
        cv2.drawContours(img,[box],0,color,thickness=2)
        #cv2.circle(img, (int(self.center[0]), int(self.center[1])), 8, color, thickness=4)
        cv2.putText(img, self.label, (int(self.center[0] - 50), int(self.center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness=2)
        #for p in self.points:
        #    _=cv2.circle(img, (int(p[0]), int(p[1])), 6, color=(255,0,255), thickness=3)

    def get_area(self):
        center, lengths, angle=self.rect
        return np.prod(lengths)        

    def __str__(self):
        return f'ViewObject: {self.label} at {self.center}'

    def get_bbox(self): #Axis-aligned bbox in image-coordinates
        box=np.int0(cv2.boxPoints(self.rect))
        return (np.min(box[:,0]), np.min(box[:,1]), np.max(box[:,0]), np.max(box[:,1]) )


    #score_color and score_corner in scene_graph_scoring!

class ViewObject_old:
    __slots__ = ['label', 'bbox', 'depth', 'center', 'color']

    @classmethod
    def from_patch(cls, patch):
        v=ViewObject()
        v.label=patch.label
        v.bbox=patch.bbox #CARE: Center is normed, BBbox is not!
        v.depth=patch.depth
        v.center=patch.center/(IMAGE_WIDHT, IMAGE_HEIGHT) #Convert center [0,IMAGE_W/H] -> [0,1]
        v.color=patch.color/255.0 #Convert color [0,255] -> [0,1]
        return v

    def __str__(self):
        return f'ViewObject: {self.color} {self.label} at {self.center}'

    def score_color(target_name):
        pass

    def score_corner(taget_name):
        pass