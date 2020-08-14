import numpy as np
import cv2
import os
import random
import pickle
from graphics.rendering import CLASSES_DICT, CLASSES_COLORS
from .clustering import ClusteredObject
from .geometry import get_camera_matrices
from semantic.geometry import IMAGE_WIDHT,IMAGE_HEIGHT
from graphics.rendering import Pose
from .patches import Patch
from .utils import draw_relationships

#Point-cloud clustering and hidden points: http://www.open3d.org/docs/release/tutorial/Basic/pointcloud.html -> Doesn't seem to work ✖
#Simple depth-map: compute distance from eye for all points, color them accordingly (scaled to max)
#Occlusion check: if 2D-overlap >0.5 of object, check if other object is closer (simple bouding boxes 2D)
#Artifacts try #2: cluster, then all inside 3d bboxes 

'''
TODO
New strategy: search for small/big enough 2D blobs (dep. on class), describe 3D-relations using depth (small-small, small-big)
check for congruent depth (small only)?
Boden nur erwähnen (left, right, mid, across)
Fit rotated rects for in-front-of / behind?

Make scoring-logic same for both styles of creation? Otherwise split this file?
'''

RELATIONSHIP_TYPES=('left','right','below','above','infront','behind')
DEPTH_DIST_FACTOR=IMAGE_WIDHT/255.0*2

class Relationship:
    def __init__(self, sub,rel_type,obj):
        pass

class Relationship2:
    __slots__ = ['sub_label', 'rel_type', 'obj_label']
    def __init__(self, sub_label, rel_type, obj_label):
        self.sub_label=sub_label
        self.rel_type=rel_type
        self.obj_label=obj_label

    def __str__(self):
        return f'Relationship2: {self.sub_label} is {self.rel_type} of {self.obj_label}'

def text_from_scenegraph(relations):
    text="There is "
    for i in range(len(relations)):
        r=relations[i]
        text+=f'{r.sub_label} {r.rel_type} of {r.obj_label}'
        if i<len(relations)-1:
            text+=" and "
        else:
            text+="."
    return text

'''
Logic via patches
'''
def get_patches_relationship(sub, obj):
    center_diff=obj.center-sub.center #sub -> obj vector
    depth_diff=obj.depth-sub.depth
    if np.abs(depth_diff)*DEPTH_DIST_FACTOR>np.linalg.norm(center_diff):
        return 'infront' if depth_diff>0 else 'behind'
    else:
        dleft= obj.bbox[0]-(sub.bbox[0]+sub.bbox[2])
        dright= sub.bbox[0]-(obj.bbox[0]+obj.bbox[2])
        dbelow= sub.bbox[1]-(obj.bbox[1]+obj.bbox[3])
        dabove= obj.bbox[1]-(sub.bbox[1]+sub.bbox[3])
        return RELATIONSHIP_TYPES[np.argmax((dleft,dright,dbelow,dabove))]

#Calculate in a way that can be 'reversed' for scoring
def get_patches_relationship2(sub, obj):
    dleft= obj.bbox[0]-(sub.bbox[0]+sub.bbox[2])
    dright= sub.bbox[0]-(obj.bbox[0]+obj.bbox[2])
    dbelow= sub.bbox[1]-(obj.bbox[1]+obj.bbox[3])
    dabove= obj.bbox[1]-(sub.bbox[1]+sub.bbox[3])
    dinfront= (obj.depth-sub.depth)*DEPTH_DIST_FACTOR
    dbehind= (sub.depth-obj.depth)*DEPTH_DIST_FACTOR
    distances = (dleft,dright,dbelow,dabove,dinfront,dbehind)
    return RELATIONSHIP_TYPES[np.argmax(distances)]

#OPTION: relate any 2 patches / different classes / closest | Currently: closest in image-plane of different class
#TODO: pull subjects evenly from classes
def scenegraph_for_view_from_patches(view_patches, max_rels=6, return_as_references=False):
    view_patches=view_patches.copy()
    #random.shuffle(view_patches) #OPTION: shuffle

    relationships=[]
    for sub in view_patches[0:max_rels]:
        obj_candidates=[ p for p in view_patches if p.label!=sub.label]
        obj_distances=[ np.linalg.norm(sub.center-obj.center) for obj in obj_candidates]
        if len(obj_candidates)==0:
            continue

        obj=obj_candidates[ np.argmin(obj_distances) ]
        #rel_type=get_patches_relationship(sub,obj)
        rel_type=get_patches_relationship2(sub,obj)

        if return_as_references:
            relationships.append(Relationship2(sub, rel_type, obj))       
        else:
            relationships.append(Relationship2(sub.label, rel_type, obj.label))       

    return relationships

#Calculate the 'inverse' of get_patches_relationship2
#CARE: hope this doesn't cause problems...
def score_triplet(sub,rel_type,obj):
    #CARE: Make sure these match!
    dleft= obj.bbox[0]-(sub.bbox[0]+sub.bbox[2])
    dright= sub.bbox[0]-(obj.bbox[0]+obj.bbox[2])
    dbelow= sub.bbox[1]-(obj.bbox[1]+obj.bbox[3])
    dabove= obj.bbox[1]-(sub.bbox[1]+sub.bbox[3])
    dinfront= (obj.depth-sub.depth)*DEPTH_DIST_FACTOR
    dbehind= (sub.depth-obj.depth)*DEPTH_DIST_FACTOR
    distances = (dleft,dright,dbelow,dabove,dinfront,dbehind)
    score= distances[RELATIONSHIP_TYPES.index(rel_type)] / np.max(distances)
    return np.clip(score,0,1)

#Returns the score and the relationships with object-references instead of label-texts
def ground_scenegraph_to_patches(relations, patches):
    MIN_SCORE=0.1 #OPTION: hardest penalty for relationship not found
    best_groundings=[None for i in range(len(relations))]
    best_scores=[MIN_SCORE for i in range(len(relations))] 

    for i_relation,relation in enumerate(relations): #Walk through relations
        subject_label, rel_type, object_label = relation.sub_label, relation.rel_type, relation.obj_label
        #Walk through all possible groundings
        for subj in [obj for obj in patches if obj.label==subject_label]: 
            for obj in [obj for obj in patches if obj.label==object_label]:
                if subj==obj: continue
                score=score_triplet(subj,rel_type,obj)
                if score>best_scores[i_relation]:
                    best_groundings[i_relation]= Relationship2(subj, rel_type, obj) #(subj,rel_type,obj)
                    best_scores[i_relation]=score

    return np.prod(best_scores), best_groundings    

#Can't assume completeness
def score_scenegraph_pair(relations0, relations1):
    pass
    

'''
Logic via 3D-Clustering
'''
#Naive approach: An object is (potentially) occluded if another object is closer to the camera and occludes more than half of it in the x-y-image-plane
def is_object_occluded(obj, scene_objects):
    for occluder in scene_objects:
        if occluder==obj:
            continue

def get_midpoint_distance(sub,obj):
    smid=0.5*(sub.bbox_projected[0:3]+sub.bbox_projected[3:6])
    omid=0.5*(obj.bbox_projected[0:3]+obj.bbox_projected[3:6])
    return np.linalg.norm(smid-omid)

def get_area(rotated_rect):
    return rotated_rect[1][0]*rotated_rect[1][1]    

#TODO: via best score or via logic?
def get_relationship_type(sub, obj):
    #TODO: intersect on floor-plane first (in world-coords!) - Not necessary anymore?
    #floor_intersection=None

    #Intersect in image-plane, if intersection > 0.5 x smaller-area -> infront/behind
    intersection_type, intersection_points=cv2.rotatedRectangleIntersection(sub.image_rect, obj.image_rect)
    if intersection_type != cv2.INTERSECT_NONE:
        intersection_area=get_area(cv2.minAreaRect(intersection_points))
        if intersection_area >= 0.5*np.minimum( get_area(sub.image_rect), get_area(obj.image_rect)):
            if sub.bbox_projected[2]<obj.bbox_projected[2]:
                return 'infront' #Subject is in front of object
            else:
                return 'behind' #Subject is behind object

    #OPTION: also allow infront/behind based on distances?
    dleft = np.min(obj.i_boxpoints[:,0]) - np.max(sub.i_boxpoints[:,0])
    dright= np.min(sub.i_boxpoints[:,0]) - np.max(obj.i_boxpoints[:,0])
    dbelow= np.min(obj.i_boxpoints[:,1]) - np.max(sub.i_boxpoints[:,1])
    dabove= np.min(sub.i_boxpoints[:,1]) - np.max(obj.i_boxpoints[:,1])
    
    return RELATIONSHIP_TYPES[ np.argmin(np.abs((dleft,dright,dbelow,dabove))) ] #Min or max?!
    

#General assumption: No 3D-intersections of objects
def annotate_view(pose, scene_objects):
    relationships=[]

    I,E=get_camera_matrices(pose)
    for obj in scene_objects:
        obj.project(I,E)

    fov_objects=[obj for obj in scene_objects if obj.in_fov()]

    visible_objects=[obj for obj in fov_objects if not is_object_occluded(obj, fov_objects)]

    for sub in visible_objects:
        closest_object=None #OPTION: closest or any visible?


'''
Data Scene-Graph creation
'''
#scene_relationships as { file_name: [rels] }
def create_scenegraphs(base_path, scene_name):
    print('Scenegraphs for scene',scene_name)
    scene_patches=pickle.load(open(os.path.join(base_path, scene_name,'patches.pkl'), 'rb'))
    scene_relationships={}

    for file_name in scene_patches.keys():
        print(file_name)
        view_relationships=scenegraph_for_view_from_patches(scene_patches[file_name], return_as_references=False)
        scene_relationships[file_name]=view_relationships
        
        # #Debugging
        # view_relationships_reference=scenegraph_for_view_from_patches(scene_patches[file_name], return_as_references=True)
        # image_rgb=cv2.imread(os.path.join(base_path, scene_name,'lbl', file_name))
        # draw_relationships(image_rgb, view_relationships_reference)
        # cv2.imshow("",image_rgb)
        # #cv2.imwrite(file_name,image_rgb)
        # #print(file_name,text_from_scenegraph(rels[0:3]))
        # cv2.waitKey()

        #break
    return scene_relationships        


if __name__ == "__main__":
    base_path='data/pointcloud_images_3_2_depth'
    scene_name='sg27_station2_intensity_rgb'

    '''
    Scene-Graph creation
    '''
    scene_relationships=create_scenegraphs(base_path, scene_name)   
    pickle.dump( scene_relationships, open(os.path.join(base_path, scene_name,'scenegraphs.pkl'), 'wb'))