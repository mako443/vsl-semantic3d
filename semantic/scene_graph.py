import numpy as np
import cv2
import os
import random
import pickle
from graphics.rendering import CLASSES_DICT, CLASSES_COLORS
from .clustering import ClusteredObject
#from .geometry import get_camera_matrices
#from semantic.geometry import IMAGE_WIDHT,IMAGE_HEIGHT
from graphics.imports import CLASSES_DICT, CLASSES_COLORS, Pose, IMAGE_WIDHT, IMAGE_HEIGHT
from .patches import Patch
from .utils import draw_relationships, draw_view_objects, draw_scenegraph, draw_patches
import semantic.scene_graph_cluster3d_scoring
from .imports import ViewObject, SceneGraph, SceneGraphObject, COLORS, COLOR_NAMES, CORNERS, CORNER_NAMES

from dataloading.data_loading import Semantic3dDataset


'''
Module to generate Scene Graphs and text descriptions from view-objects | Disregard patches-logic for now | No identities for now
'''

'''
TODO
-add "foreground/background" to corners (meaning across) ✓
-refine obj/sub selection: use unused ones, just keep track in list ✓

-remove deprecated
-resolve oblique infront/behind?
'''       

#Storing objects as references
#DEPRECATED?
# class SceneGraphRelationship:
#     __slots__ = ['sub', 'rel_type', 'obj']
#     def __init__(self, sub, rel_type, obj):
#         self.sub=sub
#         self.rel_type=rel_type
#         self.obj=obj




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

def score_color():
    pass

#def find_best_corner

#Uses new object-oriented SG model
#TODO: ok to just put in relationships 
#CARE: does not necessarily return relations with objects for each corner!
def scenegraph_for_view_5corners(view_objects, keep_viewobjects=False, flip_relations=True):
    
    scene_graph=SceneGraph()

    #Walk through all corners
    for corner_idx, corner in enumerate(CORNERS):
        #find the best-matching subject
        corner_dists=[ np.linalg.norm(sub.center - corner) for sub in view_objects ]
        sub = view_objects[ np.argmin(corner_dists) ]
        #print(sub.label, sub.center)

        #Find closest (in image-plane) object of other class; chosing other class also reduces merged/split problems
        obj_candidates=[ p for p in view_objects if p.label!=sub.label]
        obj_distances=[ np.linalg.norm(sub.center-obj.center) for obj in obj_candidates]

        if len(obj_candidates)==0:
            continue

        obj=obj_candidates[ np.argmin(obj_distances) ]
        rel_type=get_patches_relationship2(sub, obj)

        if flip_relations:
            if rel_type=='right':
                sub,rel_type,obj= obj, 'left', sub
            if rel_type=='above':
                sub,rel_type,obj= obj, 'below', sub
            if rel_type=='behind':
                sub,rel_type,obj= obj, 'infront', sub

        if keep_viewobjects: #Debugging
            scene_graph.add_relationship( sub, rel_type,  obj )
        else:
            scene_graph.add_relationship( SceneGraphObject.from_viewobject(sub), rel_type,  SceneGraphObject.from_viewobject(obj) )

    return scene_graph

def scenegraph_for_view_cluster3d_5corners(view_objects, keep_viewobjects=False, flip_relations=False):
    scene_graph=SceneGraph()

    corners={ obj : get_corner(obj) for obj in view_objects }

    for corner_idx, corner in enumerate(CORNERS):
        #find subject closest to corner
        corner_dists=[ np.linalg.norm(np.array(sub.center)/(IMAGE_WIDHT,IMAGE_HEIGHT) - corner) for sub in view_objects ]   
        sub = view_objects[ np.argmin(corner_dists) ]

        #Find closest (in image-plane) object of same or other class
        obj_candidates=[ obj for obj in view_objects if obj!=sub]   
        obj_distances=[ np.linalg.norm(np.array(sub.center)-np.array(obj.center)) for obj in obj_candidates]  

        if len(obj_candidates)==0:
            continue        

        obj=obj_candidates[ np.argmin(obj_distances) ]
        rel_type=semantic.scene_graph_cluster3d_scoring.get_relationship_type(sub, obj)

        if flip_relations:
            if rel_type=='right':
                sub,rel_type,obj= obj, 'left', sub
            if rel_type=='above':
                sub,rel_type,obj= obj, 'below', sub
            if rel_type=='behind':
                sub,rel_type,obj= obj, 'infront', sub

        if keep_viewobjects: #Debugging
            scene_graph.add_relationship( sub, rel_type,  obj )
        else:
            scene_graph.add_relationship( SceneGraphObject.from_viewobject_cluster3d(sub), rel_type,  SceneGraphObject.from_viewobject_cluster3d(obj) )

    return scene_graph  


'''
Strategy refined from 5 corners:
-Use one subject closest to each corner (can be fg/bg)
-Use unused object from same corner or fg/bg (fg/bg allowed multiple times)
'''
def scenegraph_for_view_cluster3d_7corners(view_objects, keep_viewobjects=False, flip_relations=True):
    assert len(CORNERS)==5 #FG/BG not explicitly in corners

    scene_graph=SceneGraph()
    used_objects=[]

    if len(view_objects)<2:
        print('scenegraph_for_view_cluster3d_7corners(): returning Empty Scene Graph, not enough view objects')
        return scene_graph

    for corner_idx, corner in enumerate(CORNERS):
        corner_name=CORNER_NAMES[corner_idx]

        #find subject closest to corner
        corner_dists=[ np.linalg.norm(np.array(sub.center)/(IMAGE_WIDHT,IMAGE_HEIGHT) - corner) for sub in view_objects ]   
        sub = view_objects[ np.argmin(corner_dists) ]

        #Object selection
        #Care not to add fg/bg to used_objects
        obj_candidates=[ obj for obj in view_objects if obj!=sub and SceneGraphObject.from_viewobject_cluster3d(sub).corner in (corner_name, 'foreground', 'background') and obj not in used_objects ]
        obj_distances=[ np.linalg.norm(np.array(sub.center)-np.array(obj.center)) for obj in obj_candidates]  

        if len(obj_candidates)==0:
            continue

        obj=obj_candidates[ np.argmin(obj_distances) ]
        rel_type=semantic.scene_graph_cluster3d_scoring.get_relationship_type(sub, obj)

        if SceneGraphObject.from_viewobject_cluster3d(sub).corner not in ('foreground', 'background'):
            used_objects.append(sub)
        if SceneGraphObject.from_viewobject_cluster3d(obj).corner not in ('foreground', 'background'):
            used_objects.append(obj)            
        
        if flip_relations:
            if rel_type=='right':
                sub,rel_type,obj= obj, 'left', sub
            if rel_type=='above':
                sub,rel_type,obj= obj, 'below', sub
            if rel_type=='behind':
                sub,rel_type,obj= obj, 'infront', sub

        if keep_viewobjects: #Debugging
            scene_graph.add_relationship( sub, rel_type,  obj )
        else:
            scene_graph.add_relationship( SceneGraphObject.from_viewobject_cluster3d(sub), rel_type,  SceneGraphObject.from_viewobject_cluster3d(obj) )

    return scene_graph  



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

#Can't assume completeness, does this even make sense?
def score_scenegraph_pair(relations0, relations1):
    pass

'''
Logic via 3D-Clustering
'''

def get_area(rotated_rect):
    return rotated_rect[1][0]*rotated_rect[1][1]    

# #TODO: via best score or via logic?
# def get_relationship_type(sub, obj):
#     #TODO: intersect on floor-plane first (in world-coords!) - Not necessary anymore?
#     #floor_intersection=None

#     #Intersect in image-plane, if intersection > 0.5 x smaller-area -> infront/behind
#     intersection_type, intersection_points=cv2.rotatedRectangleIntersection(sub.image_rect, obj.image_rect)
#     if intersection_type != cv2.INTERSECT_NONE:
#         intersection_area=get_area(cv2.minAreaRect(intersection_points))
#         if intersection_area >= 0.5*np.minimum( get_area(sub.image_rect), get_area(obj.image_rect)):
#             if sub.bbox_projected[2]<obj.bbox_projected[2]:
#                 return 'infront' #Subject is in front of object
#             else:
#                 return 'behind' #Subject is behind object

#     #OPTION: also allow infront/behind based on distances?
#     dleft = np.min(obj.i_boxpoints[:,0]) - np.max(sub.i_boxpoints[:,0])
#     dright= np.min(sub.i_boxpoints[:,0]) - np.max(obj.i_boxpoints[:,0])
#     dbelow= np.min(obj.i_boxpoints[:,1]) - np.max(sub.i_boxpoints[:,1])
#     dabove= np.min(sub.i_boxpoints[:,1]) - np.max(obj.i_boxpoints[:,1])
    
#     return RELATIONSHIP_TYPES[ np.argmin(np.abs((dleft,dright,dbelow,dabove))) ] #Min or max?!
    

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
Data creation: Scene-Graphs
'''
#scene_relationships as { file_name: [rels] }
def create_scenegraphs(base_path, scene_name):
    print('Scenegraphs for scene',scene_name)
    scene_patches=pickle.load(open(os.path.join(base_path, scene_name,'view_objects.pkl'), 'rb'))
    scene_graphs={}

    for file_name in scene_patches.keys():
        print(f'\r {file_name}', end='')
        view_relationships=scenegraph_for_view_cluster3d_7corners(scene_patches[file_name], keep_viewobjects=False)
        scene_graphs[file_name]=view_relationships
        
        # #Debugging
        # view_relationships_reference=scenegraph_for_view_from_patches(scene_patches[file_name], return_as_references=True)
        # image_rgb=cv2.imread(os.path.join(base_path, scene_name,'lbl', file_name))
        # draw_relationships(image_rgb, view_relationships_reference)
        # cv2.imshow("",image_rgb)
        # #cv2.imwrite(file_name,image_rgb)
        # #print(file_name,text_from_scenegraph(rels[0:3]))
        # cv2.waitKey()

        #break
    print()
    return scene_graphs     


if __name__ == "__main__":
    ### Scene graph debugging for Cluster3d
    base_path='data/pointcloud_images_o3d_merged/'
    scene_name='bildstein_station1_xyz_intensity_rgb'
    #scene_name=np.random.choice(('domfountain_station1_xyz_intensity_rgb','sg27_station2_intensity_rgb','untermaederbrunnen_station1_xyz_intensity_rgb','neugasse_station1_xyz_intensity_rgb'))
    scene_view_objects=pickle.load( open(os.path.join(base_path,scene_name,'view_objects.pkl'), 'rb') )

    file_name='012.png'
    #file_name=np.random.choice(list(scene_view_objects.keys()))
    view_objects=scene_view_objects[file_name]
    print(f'{scene_name} - {file_name} {len(view_objects)} view objects')

    texts=[ str(SceneGraphObject.from_viewobject_cluster3d(v)) for v in view_objects ]
    print(texts)
    print()

    sg=scenegraph_for_view_cluster3d_7corners(view_objects, keep_viewobjects=False)
    print(sg.get_text())
    score, groundings= semantic.scene_graph_cluster3d_scoring.score_sceneGraph_to_viewObjects(sg, view_objects)
    print('SG-Score:',score)

    sg=scenegraph_for_view_cluster3d_7corners(view_objects, keep_viewobjects=True)

    img=cv2.imread(os.path.join(base_path, scene_name,'rgb', file_name))
    draw_view_objects(img, view_objects, texts)    
    cv2.imshow("",img); cv2.waitKey()

    img=cv2.imread(os.path.join(base_path, scene_name,'rgb', file_name))
    draw_scenegraph(img,sg)
    cv2.imshow("",img); cv2.waitKey()
    
    img=cv2.imread(os.path.join(base_path, scene_name,'rgb', file_name))
    draw_scenegraph(img,groundings)
    cv2.imshow("",img); cv2.waitKey()
    
    cv2.imwrite("sg_demo.jpg",img)
    quit()
    ### Scene graph debugging for Cluster3d

    ### Scene Graph Eval debugging
    dataset=Semantic3dDataset('data/pointcloud_images_o3d')
    idx0=10
    idx1=100
    sg=dataset.view_scenegraphs[idx0]
    img_sg=cv2.imread(dataset.image_paths[idx0])
    img_test=cv2.imread(dataset.image_paths[idx1])
    print(sg.get_text())
    print()
    cv2.imshow("",img_sg); cv2.waitKey()

    score, groundings= semantic.scene_graph_cluster3d_scoring.score_sceneGraph_to_viewObjects(sg, dataset.view_objects[idx1])
    groundings=[g for g in groundings if g is not None]
    sub,rel_type,obj=groundings[-1]
    print(sub.label, rel_type, obj.label)
    print(sub.mindist,sub.maxdist,sub.get_bbox())
    print(obj.mindist,obj.maxdist,obj.get_bbox())
    s=semantic.scene_graph_cluster3d_scoring.score_relationship(sub, rel_type, obj, output_print=True)
    print(s)

    print('SG-Score:',score)
    draw_scenegraph(img_test,[g for g in groundings if g is not None] )
    cv2.imshow("",img_test); cv2.waitKey()


    quit()
    ### Scene Graph Eval debugging

    '''
    Data creation: Scene-Graphs from view-objects
    '''
    base_path='data/pointcloud_images_o3d'
    for scene_name in ('domfountain_station1_xyz_intensity_rgb','sg27_station2_intensity_rgb','untermaederbrunnen_station1_xyz_intensity_rgb','neugasse_station1_xyz_intensity_rgb'):
        scene_graphs=create_scenegraphs(base_path, scene_name)   
        pickle.dump( scene_graphs, open(os.path.join(base_path, scene_name,'scene_graphs.pkl'), 'wb'))