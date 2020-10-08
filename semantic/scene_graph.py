import numpy as np
import cv2
import os
import random
import pickle
from graphics.rendering import CLASSES_DICT, CLASSES_COLORS
from .clustering import ClusteredObject
#from .geometry import get_camera_matrices
#from semantic.geometry import IMAGE_WIDHT,IMAGE_HEIGHT
from graphics.imports import CLASSES_DICT, CLASSES_COLORS, Pose, IMAGE_WIDHT, IMAGE_HEIGHT, COMBINED_SCENE_NAMES
from .patches import Patch
from .utils import draw_relationships, draw_view_objects, draw_scenegraph, draw_patches
import semantic.scene_graph_cluster3d_scoring
from .imports import ViewObject, SceneGraph, SceneGraphObject, COLORS, COLOR_NAMES, CORNERS, CORNER_NAMES

from dataloading.data_loading import Semantic3dDataset
from .scene_graph_cluster3d_scoring import get_relationship_type, score_relationship_type


'''
Module to generate Scene Graphs and text descriptions from view-objects | Disregard patches-logic for now | No identities for now
'''

'''
TODO
-add "foreground/background" to corners (meaning across) ✓
-refine obj/sub selection: use unused ones, just keep track in list ✓
-add & score strategy with closest relations ✓

-remove deprecated
-resolve oblique infront/behind?
-add closest/furthest attributes
'''       

'''
Logic via patches
'''
# def get_patches_relationship(sub, obj):
#     center_diff=obj.center-sub.center #sub -> obj vector
#     depth_diff=obj.depth-sub.depth
#     if np.abs(depth_diff)*DEPTH_DIST_FACTOR>np.linalg.norm(center_diff):
#         return 'infront' if depth_diff>0 else 'behind'
#     else:
#         dleft= obj.bbox[0]-(sub.bbox[0]+sub.bbox[2])
#         dright= sub.bbox[0]-(obj.bbox[0]+obj.bbox[2])
#         dbelow= sub.bbox[1]-(obj.bbox[1]+obj.bbox[3])
#         dabove= obj.bbox[1]-(sub.bbox[1]+sub.bbox[3])
#         return RELATIONSHIP_TYPES[np.argmax((dleft,dright,dbelow,dabove))]

# #Calculate in a way that can be 'reversed' for scoring
# def get_patches_relationship2(sub, obj):
#     dleft= obj.bbox[0]-(sub.bbox[0]+sub.bbox[2])
#     dright= sub.bbox[0]-(obj.bbox[0]+obj.bbox[2])
#     dbelow= sub.bbox[1]-(obj.bbox[1]+obj.bbox[3])
#     dabove= obj.bbox[1]-(sub.bbox[1]+sub.bbox[3])
#     dinfront= (obj.depth-sub.depth)*DEPTH_DIST_FACTOR
#     dbehind= (sub.depth-obj.depth)*DEPTH_DIST_FACTOR
#     distances = (dleft,dright,dbelow,dabove,dinfront,dbehind)
#     return RELATIONSHIP_TYPES[np.argmax(distances)]

'''
Logic via Cluster3D
'''

'''
Strategy refined from 5 corners:
-Use one subject closest to each corner (can be fg/bg)
-Use unused object from same corner or fg/bg (fg/bg allowed multiple times)
=> Used for text generation
'''
#TODO: New strat: Use biggest (in image plane) subject from each corner (if any), relate to closest object
#OR
#TODO: New strat: Get biggest (in image plane) subject for each corner (if any), relate to closest in corner (if any), mention fg&bg object
#TODO: Split SGs and Captions completely? Which one used for Geometric? (Want both?)
def scenegraph_for_view_cluster3d_7corners(view_objects, keep_viewobjects=False, flip_relations=True):
    assert len(CORNERS)==5 #FG/BG not explicitly in corners

    scene_graph=SceneGraph()
    used_objects=[]

    if len(view_objects)<2:
        #print('scenegraph_for_view_cluster3d_7corners(): returning Empty Scene Graph, not enough view objects')
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

'''
Strategy: Create a relationship for each object w/ its nearest neighbor (no doublicates)
=> Used for SG matching
'''
def scenegraph_for_view_cluster3d_nnRels(view_objects, keep_viewobjects=False, flip_relations=True):
    assert len(CORNERS)==5 #FG/BG not explicitly in corners

    scene_graph=SceneGraph()
    blocked_subjects=[]

    if len(view_objects)<2:
        #print('scenegraph_for_view_cluster3d_nnRels(): returning Empty Scene Graph, not enough view objects')
        return scene_graph

    for sub in view_objects:
        if sub in blocked_subjects: continue

        min_dist=np.inf
        min_obj=None
        for obj in view_objects:
            if sub is obj: continue

            dist=np.linalg.norm(sub.get_center_c_world() - obj.get_center_c_world())
            if dist<min_dist:
                min_dist=dist
                min_obj=obj
        
        obj=min_obj
        assert obj is not None
        rel_type=semantic.scene_graph_cluster3d_scoring.get_relationship_type(sub, obj)
        
        blocked_subjects.append(obj) #No doublicate relations, CARE: do before flip

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

#Can't assume completeness, does this even make sense? 
#TODO: for text-pairs? For SG<->SG matching? 
#CARE: only makes sense if all objects used
def score_scenegraph_pair(relations0, relations1):
    pass

def get_area(rotated_rect):
    return rotated_rect[1][0]*rotated_rect[1][1]    


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
Data creation: Scene-Graphs and Captions
'''
#scene_relationships as { file_name: [rels] }
def create_scenegraphs_nnRels(base_path,split, scene_name):
    print('Scenegraphs (for matching) for scene',scene_name)
    scene_view_objects=pickle.load(open(os.path.join(base_path,split, scene_name,'view_objects.pkl'), 'rb'))
    scene_graphs={}

    for file_name in scene_view_objects.keys():
        print(f'\r {file_name}', end='')
        view_scenegraph=scenegraph_for_view_cluster3d_nnRels(scene_view_objects[file_name], keep_viewobjects=False)
        scene_graphs[file_name]=view_scenegraph
        
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

#DEPRECATED | more empty and less-rich captions
def create_captions_7corners(base_path,split, scene):
    print('Scenegraphs (for captions) for scene',scene_name)
    scene_view_objects=pickle.load(open(os.path.join(base_path,split, scene_name,'view_objects.pkl'), 'rb'))
    captions={}

    for file_name in scene_view_objects.keys():
        print(f'\r {file_name}', end='')
        view_scenegraph=scenegraph_for_view_cluster3d_7corners(scene_view_objects[file_name], keep_viewobjects=False)
        captions[file_name]=view_scenegraph.get_text()
        
    print()
    return captions     

def create_captions_nnRels(base_path,split, scene):
    print('Scenegraphs (for captions, NN-Rels) for scene',scene_name)
    scene_view_objects=pickle.load(open(os.path.join(base_path,split, scene_name,'view_objects.pkl'), 'rb'))
    captions={}

    for file_name in scene_view_objects.keys():
        print(f'\r {file_name}', end='')
        view_scenegraph=scenegraph_for_view_cluster3d_nnRels(scene_view_objects[file_name], keep_viewobjects=False)
        captions[file_name]=view_scenegraph.get_text_extensive()
        
    print()
    return captions      


if __name__ == "__main__":
    ## Scene graph debugging for Cluster3d
    # base_path='data/pointcloud_images_o3d_merged/'
    # scene_name='bildstein_station1_xyz_intensity_rgb'
    # split='test'
    # scene_view_objects=pickle.load( open(os.path.join(base_path,split,scene_name,'view_objects.pkl'), 'rb') )

    # file_name='003.png'
    # view_objects=scene_view_objects[file_name]
    # print(f'{scene_name} - {file_name} {len(view_objects)} view objects')
    # sg0=scenegraph_for_view_cluster3d_nnRels(view_objects, keep_viewobjects=False, flip_relations=False)
    # sg0_debug=scenegraph_for_view_cluster3d_nnRels(view_objects, keep_viewobjects=True, flip_relations=False)
    # img=cv2.imread(os.path.join(base_path,split, scene_name,'rgb', file_name))
    # draw_scenegraph(img,sg0_debug)
    # #draw_view_objects(img, view_objects, [o.label for o in view_objects])
    # #cv2.imshow(file_name,img)
    # objs=semantic.scene_graph_cluster3d_scoring.extract_scenegraph_objects(sg0)
    # for o in objs: print(o)

    # file_name='009.png'
    # view_objects=scene_view_objects[file_name]
    # print(f'{scene_name} - {file_name} {len(view_objects)} view objects')
    # sg1=scenegraph_for_view_cluster3d_nnRels(view_objects, keep_viewobjects=False, flip_relations=False)
    # sg1_debug=scenegraph_for_view_cluster3d_nnRels(view_objects, keep_viewobjects=True, flip_relations=False)
    # img=cv2.imread(os.path.join(base_path,split, scene_name,'rgb', file_name))
    # draw_scenegraph(img,sg1_debug)
    # cv2.imshow(file_name,img)

    # print(semantic.scene_graph_cluster3d_scoring.score_sceneGraph_to_sceneGraph_nnRels(sg0,sg0))
    # print(semantic.scene_graph_cluster3d_scoring.score_sceneGraph_to_sceneGraph_nnRels(sg1,sg1))
    # print()
    # print(semantic.scene_graph_cluster3d_scoring.score_sceneGraph_to_sceneGraph_nnRels(sg0,sg1))
    # print(semantic.scene_graph_cluster3d_scoring.score_sceneGraph_to_sceneGraph_nnRels(sg1,sg0))
    
    # #cv2.waitKey()    

    # # print(sg.get_text())
    # # score= semantic.scene_graph_cluster3d_scoring.scenegraph_similarity(sg,sg)
    # # print('SG-Score:',score)

    # # img=cv2.imread(os.path.join(base_path,split, scene_name,'rgb', file_name))
    # # draw_scenegraph(img,sg_debug)
    # # cv2.imshow("",img); cv2.waitKey()
    # quit()
    ## Scene graph debugging for Cluster3d

    ### Scene graph nnRels debugging
    # base_path='data/pointcloud_images_o3d_merged/'
    # scene_name='bildstein_station1_xyz_intensity_rgb'
    # split='train'
    # scene_view_objects=pickle.load( open(os.path.join(base_path,split,scene_name,'view_objects.pkl'), 'rb') )
    # file_name=np.random.choice(list(scene_view_objects.keys()))
    # #file_name='005.png'
    # view_objects=scene_view_objects[file_name]
    # print(f'{scene_name} - {file_name} {len(view_objects)} view objects')

    # sg=scenegraph_for_view_cluster3d_nnRels(view_objects, keep_viewobjects=False, flip_relations=False)
    # print(sg.get_text_extensive())

    # sg_debug=scenegraph_for_view_cluster3d_nnRels(view_objects, keep_viewobjects=True, flip_relations=False)

    # img=cv2.imread(os.path.join(base_path,split, scene_name,'rgb', file_name))
    # draw_scenegraph(img,sg_debug)
    # cv2.imshow("",img); cv2.waitKey()

    # score, grounding=semantic.scene_graph_cluster3d_scoring.score_sceneGraph_to_viewObjects_nnRels(sg, view_objects)
    # print('Score',score)

    # # img=cv2.imread(os.path.join(base_path, scene_name,'rgb', file_name))
    # # draw_scenegraph(img,grounding)
    # # cv2.imshow("",img); cv2.waitKey()  

    # # used_objects=[]
    # # for rel in sg_debug.relationships:
    # #     if rel[0] not in used_objects: used_objects.append(rel[0])
    # #     if rel[2] not in used_objects: used_objects.append(rel[2])              
    
    # # score, groundings, unused_obects=semantic.scene_graph_cluster3d_scoring.score_sceneGraph_to_viewObjects_nnRels(sg, view_objects, unused_factor=True)
    # # print('Score w/ unused:',score)

    # quit()
    ### Scene graph nnRels debugging

    ### Scene graph nnRels scoring debugging
    # base_path='data/pointcloud_images_o3d_merged/'
    # scene_name='sg27_station1_intensity_rgb'
    # scene_view_objects=pickle.load( open(os.path.join(base_path,scene_name,'view_objects.pkl'), 'rb') )
    # #file_name=np.random.choice(list(scene_view_objects.keys()))
    # file_name_query='009.png'
    # file_name_db='012.png'
    # query_objects=scene_view_objects[file_name_query]
    # db_objects=scene_view_objects[file_name_db]
    # print(f'{scene_name} - {file_name_query}->{file_name_db}; {len(db_objects)} view objects')

    # sg=scenegraph_for_view_cluster3d_nnRels(query_objects, keep_viewobjects=False, flip_relations=False)
    # sg_debug=scenegraph_for_view_cluster3d_nnRels(query_objects, keep_viewobjects=True, flip_relations=False)

    # img=cv2.imread(os.path.join(base_path, scene_name,'rgb', file_name_query))
    # draw_scenegraph(img,sg_debug)
    # cv2.imshow("",img); cv2.waitKey()

    # score, grounding=semantic.scene_graph_cluster3d_scoring.score_sceneGraph_to_viewObjects_nnRels(sg, db_objects, unused_penalty=False)
    # print('Score w/o unused',score)
    # img=cv2.imread(os.path.join(base_path, scene_name,'rgb', file_name_db))
    # draw_scenegraph(img,grounding)
    # cv2.imshow("",img); cv2.waitKey()            
    
    # score, groundings=semantic.scene_graph_cluster3d_scoring.score_sceneGraph_to_viewObjects_nnRels(sg, db_objects, unused_penalty=True)
    # print('Score w/ unused:',score)
    # img=cv2.imread(os.path.join(base_path, scene_name,'rgb', file_name_db))
    # draw_scenegraph(img,grounding)
    # #for u in unused_objects:
    #     #u.draw_on_image(img)
    # cv2.imshow("",img); cv2.waitKey() 

    # quit()
    ### Scene graph nnRels scoring debugging

    ### Scene Graph Scoring debugging
    # dataset=Semantic3dDataset('data/pointcloud_images_o3d_merged')
    # idx0=10
    # idx1=100
    # sg=dataset.view_scenegraphs[idx0]
    # img_sg=cv2.imread(dataset.image_paths[idx0])
    # img_test=cv2.imread(dataset.image_paths[idx1])
    # print(sg.get_text())
    # print()
    # cv2.imshow("",img_sg); cv2.waitKey()

    # draw_view_objects(img_test, dataset.view_objects[idx1])
    # cv2.imshow("",img_test); cv2.waitKey()
    # quit()

    # score, groundings= semantic.scene_graph_cluster3d_scoring.score_sceneGraph_to_viewObjects(sg, dataset.view_objects[idx1])
    # groundings=[g for g in groundings if g is not None]
    # draw_scenegraph(img_sg,[g for g in groundings if g is not None] )
    # print('SG-Score:',score)

    # draw_scenegraph(img_test, groundings)
    # cv2.imshow("",img_test); cv2.waitKey()


    # quit()
    ### Scene Graph Scoring debugging

    '''
    Data creation: Scene-Graphs and Captions from view-objects (separate SG strategies)
    '''
    base_path='data/pointcloud_images_o3d_merged_occ'   
    for split in ('train','test',):  
        for scene_name in COMBINED_SCENE_NAMES:
            print(f'\n\n Scene-Graphs and Captions for scene <{scene_name}> split <{split}>')
            scene_graphs=create_scenegraphs_nnRels(base_path,split,scene_name)   
            pickle.dump( scene_graphs, open(os.path.join(base_path,split, scene_name,'scene_graphs.pkl'), 'wb'))

            #scene_captions=create_captions_7corners(base_path,split,scene_name)
            scene_captions=create_captions_nnRels(base_path,split,scene_name)
            pickle.dump( scene_captions, open(os.path.join(base_path,split, scene_name,'captions.pkl'), 'wb'))
    