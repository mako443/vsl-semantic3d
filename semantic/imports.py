import numpy as np
import os
import cv2
from graphics.imports import IMAGE_WIDHT, IMAGE_HEIGHT, CLASSES_COLORS

RELATIONSHIP_TYPES=('left','right','below','above','infront','behind')
INVERSE_RELATIONSHIP_TYPES={'left':'right','right':'left','below':'above','above':'below','infront':'behind','behind':'infront'}
DEPTH_DIST_FACTOR=IMAGE_WIDHT/255.0*2

#COLOR_NAMES=('black','red','green','blue','cyan','yellow', 'purple', 'white', 'gray') #Colors as 8 corners of unit-cube plus gray
#COLORS=np.array(( (0.1,0.1,0.1), (0.6,0.1,0.1), (0.1,0.6,0.1), (0.1,0.1,0.6), (0.1,0.6,0.6), (0.6,0.6,0.1), (0.6,0.1,0.6), (0.6,0.6,0.6), (0.3,0.3,0.3) )).reshape((9,3))
#COLORS=np.array(( (0,0,0), (1,0,0), (0,1,0), (0,0,1), (0,1,1), (1,1,0), (1,0,1), (1,1,1), (0.5,0.5,0.5) )).reshape((9,3))

#From KMeans clustering, unfortunately, the colors are all gray...
COLOR_NAMES=('brightness-0','brightness-1','brightness-2','brightness-3','brightness-4','brightness-5','brightness-6','brightness-7')
COLORS=np.array([[0.15136254, 0.12655825, 0.12769653],
                [0.22413703, 0.19569607, 0.20007613],
                [0.29251393, 0.2693559 , 0.27813852],
                [0.35667216, 0.3498905 , 0.36508256],
                [0.45776146, 0.39058182, 0.38574897],
                [0.45337288, 0.46395565, 0.47795434],
                [0.52570801, 0.53530194, 0.56404256],
                [0.66988167, 0.6804131 , 0.71069241]])


CORNER_NAMES=('top-left','top-right','bottom-left','bottom-right','center') #Additionally: 'foreground', 'background'
CORNERS=np.array(( (0.2, 0.2), (0.8,0.2), (0.2,0.8), (0.8,0.8), (0.5,0.5) )).reshape((5,2)) #Corners as relative (x,y) positions

# def project_point(I,E,point):
#     point=I@E@np.hstack((point,1))
#     return np.array(( IMAGE_WIDHT-point[0]/point[2], point[1]/point[2], -point[2] )) 

# #CARE: Open3D flips axes in the E matrix when setting the Pose! -> x-flip necessary here
# def project_point(I,E,point):
#     p= E@np.hstack((point,1))
#     p= I@p[0:3]
#     return np.array(( IMAGE_WIDHT-p[0]/p[2], p[1]/p[2], p[2]))
#     #return np.array(( p[0]/p[2], p[1]/p[2], p[2]))

#CARE: Open3D flips axes in the E matrix when setting the Pose! -> x-flip necessary here
def project_points(I,E,points):
    p= np.hstack(( points, np.ones((len(points),1)) ))
    p= np.dot(p, E.T) #E@p for each point p
    p= np.dot(p[:,0:3], I.T) #I@p for each point in p
    p[:,0]= IMAGE_WIDHT-p[:,0]/p[:,2]
    p[:,1]= p[:,1]/p[:,2]
    return p

# def project_point_extrinsic(E,point):
#     p= E@np.hstack((point,1))
#     return np.array(( -p[0]/p[2],-p[1]/p[2],p[2] )) #x/y image plane, z distance
#     #return np.array(( -p[0],-p[1],p[2] )) #x/y image plane, z distance

#Returns the point in camera-coordinates but world-units (not pixels)
def project_points_extrinsic(E,points):
    p= np.hstack(( points, np.ones((len(points),1)) ))
    p= np.dot(p, E.T) #E@p for each point p
    p[:,0]= -p[:,0]/p[:,2]
    p[:,1]= -p[:,1]/p[:,2]
    return p[:,0:3]

class ClusteredObject:
    def __init__(self, scene_name, label, points_w, total_points, pointIDs, color):
        self.scene_name=scene_name
        self.label=label
        self.points_w=points_w #3D world-coordinate points, possibly reduced | convex-hull not possible because of projection errors
        #TODO: give global (for scene) indices here (from clustering.py) -> check which made it to projection & pass to View-Object -> check which are confirmed (lbl&depth)

        self.points_i=None #3D image-coordinate points, available after self.project()
        self.points_c=None
        self.rect_i=None #Rotated bounding-box in image-coords, available after self.project(), cv2.RotatedRect object
        self.mindist_i, self.maxdist_i= None, None #Distance in image-coords, available after self.project()
        
        self.total_points=total_points #Original number of points (as opposed to hull)
        self.color=color #Color as [r,g,b] in [0,1]
        
        self.pointIDs_w=pointIDs
        self.pointIDs_i=None #Available after self.project()


    def __str__(self):
        return f'ClusteredObject: {self.label} at {np.mean(self.points_w, axis=0)}, {self.total_points} points'

    def project(self, I, E):
        #points_i= np.array( [ project_point(I,E, point) for point in self.points_w ] ) #Causes projection instabilities when points are out of FoV
        points_i=project_points(I,E,self.points_w)
        #points_c= np.array( [ project_point_extrinsic(E, point) for point in self.points_w ] ) #TODO: are points_c unstable, too?
        points_c=project_points_extrinsic(E, self.points_w)
        
        mask=np.bitwise_and.reduce(( points_i[:,0]>=0, points_i[:,0]<IMAGE_WIDHT-1, points_i[:,1]>=0, points_i[:,1]<IMAGE_HEIGHT-1, points_i[:,2]>0  )) #Mask to clamp to visible region
        points_i=points_i[mask, :].copy()
        points_c=points_c[mask, :].copy()
        if len(points_i)>4: #Projection sucessful, partly in fov, CARE: requiring at least for points, otherwise buggy
            self.rect_i=cv2.minAreaRect(points_i[:,0:2].astype(np.float32))
            self.mindist_i, self.maxdist_i = np.min(points_i[:,2]), np.max(points_i[:,2])
            self.points_i=points_i
            self.points_c=points_c
            self.pointIDs_i=self.pointIDs_w[mask]
        else: #Projection failed, none in FoV
            self.rect_i=None
            self.mindist_i, self.maxdist_i= None, None
            self.pointIDs_i=None

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
    __slots__ = ['scene_name', 'label', 'points','rect', 'mindist', 'maxdist', 'center', 'color','min_z_w','max_z_w','bbox_c','center_c','lengths_c','points_c','point_ids','point_ids_visible']

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
        v.points=clustered_object.points_i #Projected image-coord points, clamped to FoV
        v.rect=clustered_object.rect_i
        v.mindist=clustered_object.mindist_i
        v.maxdist=clustered_object.maxdist_i
        v.min_z_w, v.max_z_w= np.min(clustered_object.points_w[:,2]), np.max(clustered_object.points_w[:,2]) #DEPRECATED if points_c works
        v.center=np.array(v.rect[0])
        v.color=clustered_object.color #Color as [r,g,b] in [0,1]
        #v.corner=corner #TODO: Add here!
        #v.points_c=clustered_object.points_c
        # x/y image plane, z distance, (xmin, ymin, zmin, xmax, ymax, zmax)
        v.bbox_c=np.array(( np.min(clustered_object.points_c[:,0]), np.min(clustered_object.points_c[:,1]), np.min(clustered_object.points_c[:,2]), 
                   np.max(clustered_object.points_c[:,0]), np.max(clustered_object.points_c[:,1]), np.max(clustered_object.points_c[:,2]) ))
        v.lengths_c=np.array(( v.bbox_c[3]-v.bbox_c[0], v.bbox_c[4]-v.bbox_c[1], v.bbox_c[5]-v.bbox_c[2] ))
        v.center_c=np.array(( v.bbox_c[0]+ 0.5*v.lengths_c[0], v.bbox_c[1]+ 0.5*v.lengths_c[1], v.bbox_c[2]+ 0.5*v.lengths_c[2]))
        
        assert clustered_object.pointIDs_i is not None
        v.point_ids=clustered_object.pointIDs_i
        return v

    def set_visible_points(self,mask):
        assert len(mask)==len(self.point_ids)
        self.point_ids_visible=self.point_ids[mask]

    def draw_on_image(self,img,draw_red=None):
        box=np.int0(cv2.boxPoints(self.rect))
        color=CLASSES_COLORS[self.label] if draw_red is None else (0,0,255)
        cv2.drawContours(img,[box],0,color,thickness=2)
        #cv2.circle(img, (int(self.center[0]), int(self.center[1])), 8, color, thickness=4)
        cv2.putText(img, self.label, (int(self.center[0] - 50), int(self.center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness=2)
        
        # color=CLASSES_COLORS[self.label]
        # for p in self.points[::100]:
        #     _=cv2.circle(img, (int(p[0]), int(p[1])), 2, color=(color[2],color[1],color[0]), thickness=2)
        #     cv2.putText(img, str(p[2]), (int(p[0])+2, int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thickness=1)

    def get_area(self):
        center, lengths, angle=self.rect
        return np.prod(lengths)        

    def __str__(self):
        return f'ViewObject: {self.color} {self.label} at {self.center}'

    def get_bbox(self): #Axis-aligned bbox in image-coordinates
        box=np.int0(cv2.boxPoints(self.rect))
        return (np.min(box[:,0]), np.min(box[:,1]), np.max(box[:,0]), np.max(box[:,1]) )

    #center_c but without z-division
    def get_center_c_world(self):
        return np.array(( self.center_c[0]*self.center_c[2], self.center_c[1]*self.center_c[2], self.center_c[2] ))

    # def get_bbox_c(self):
    #     return np.array(( np.min(self.points_c[:,0]),np.min(self.points_c[:,1]), np.max(self.points_c[:,0]), np.max(self.points_c[:,1]) ))


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

#Retain possibility for pure Graph-structure for potential Graph-networks later on
class SceneGraph:
    def __init__(self):
        self.relationships=[]

    #Store relationship as (SG-object, rel_type, SG-object) triplet
    def add_relationship(self, sub, rel_type, obj):
        self.relationships.append( (sub,rel_type,obj) ) 

    def get_text(self):
        if len(self.relationships)==0: #TODO/CARE: this ok?
            return 'There is nothing to describe.'

        text=''
        for rel in self.relationships:
            sub, rel_type, obj=rel
            rel_text=f'In the {sub.corner} there is a {sub.color} {sub.label} that is {rel_type} of a {obj.color} {obj.label}. '
            if rel_text not in text: #Prevent doublicate sentences
                text+=rel_text

        return text

    #Same as above, but also describes the objects corner and allows dublicates, making it equivalent to the Scene-Graph relationships used in Geometric Learning.
    def get_text_extensive(self):
        if len(self.relationships)==0: #TODO/CARE: this ok?
            return 'There is nothing to describe.'

        text=''
        for rel in self.relationships:
            sub, rel_type, obj=rel
            rel_text=f'In the {sub.corner} there is a {sub.color} {sub.label} that is {rel_type} of a {obj.color} {obj.label} in the {obj.corner}. '
           
            #Doublicates barely seem to happen.
            #if rel_text not in text: #Prevent doublicate sentences
            #    text+=rel_text

            text+=rel_text

        return text
    

    def is_empty(self):
        return len(self.relationships)==0

#TODO: attributes here or in graph? Should be possible to convert to graph
class SceneGraphObject:
    __slots__ = ['label', 'color', 'corner','maxdist','corner5','center_hash']

    # @classmethod
    # def from_viewobject(cls, v : ViewObject):
    #     sgo=SceneGraphObject()
    #     sgo.label=v.label

    #     color_distances= np.linalg.norm( COLORS-v.color, axis=1 )
    #     sgo.color=COLOR_NAMES[ np.argmin(color_distances) ]

    #     corner_distances= np.linalg.norm( CORNERS-v.center, axis=1 )
    #     sgo.corner= CORNER_NAMES[ np.argmin(corner_distances) ]

    #     return sgo

    #CARE: Potentially some differences!
    @classmethod
    def from_viewobject_cluster3d(cls, v):
        sgo=SceneGraphObject()
        sgo.label=v.label

        color_distances= np.linalg.norm( COLORS-v.color, axis=1 )
        sgo.color=COLOR_NAMES[ np.argmin(color_distances) ]

        corner_distances= np.linalg.norm( CORNERS- (v.center/(IMAGE_WIDHT, IMAGE_HEIGHT)), axis=1 )
        sgo.corner5= CORNER_NAMES[ np.argmin(corner_distances) ] #Corner w/o fg/bg, only used in SG-SG scoring
    
        #TODO: score corner (always possible to score as fg/bg or one of the corners?)
        if np.max(v.rect[1])>=2/3*IMAGE_WIDHT:
            sgo.corner='foreground' if v.center[1]>IMAGE_HEIGHT/2 else 'background'
        else:
            sgo.corner=sgo.corner5

        #sgo.maxdist=v.maxdist
        sgo.center_hash=v.center #Center of View-Object, only used to compare identities in co-reference ablation study

        return sgo
    

    def __str__(self):
        return f'{self.color} {self.label} at {self.corner}'

    def get_text(self):
        return str(self)             