import numpy as np
from graphics.imports import CLASSES_DICT, CLASSES_COLORS, IMAGE_WIDHT, IMAGE_HEIGHT
from .imports import SceneGraph,SceneGraphObject, ViewObject, COLORS, COLOR_NAMES, CORNERS, CORNER_NAMES, RELATIONSHIP_TYPES

'''
Create & Score Strategies
- Distant edges: only fail for full intersection, leads to wrong text descriptions
- Invert for full overlap: only works with distant edges -> also wrong texts
- Weighted center differences: Weight with max size in that direction <== Risk this one
'''

#TODO: how to resolve the 'terrain angled in front of hard-scape' situation?
#Select the relationship type as the direction with the largest distance
#CARE: Comparing in different Units, pixels vs. world-coordinates!!
# def get_relationship_type(sub : ViewObject, obj : ViewObject):
#     sub_bbox, obj_bbox=sub.get_bbox(), obj.get_bbox()
#     #Compare left/right in image-coords #is this really the same world-coords in points_c?
#     dleft= obj_bbox[0] - sub_bbox[2]
#     dright= sub_bbox[0] - obj_bbox[2]
#     #Compare below/above in world-coords
#     dbelow= obj.min_z_w - sub.max_z_w
#     dabove= sub.min_z_w - obj.max_z_w
#     dinfront= obj.mindist - sub.maxdist
#     dbehind= sub.mindist - obj.maxdist

#     #With too much overlap, it can only be in front or behind #CARE: this is not reversible #TODO: not necessary w/ world-units?
#     if np.sum(( dleft<0, dright<0, dbelow<0, dabove<0 ))>=3:
#         dleft, dright=-1e6,-1e6
#         dbelow, dabove=-1e6,-1e6
#         # #sub_center_z=1/2*(np.min(sub.points[:,2])+ np.max(sub.points[:,2]))
#         # #obj_center_z=1/2*(np.min(obj.points[:,2])+ np.max(obj.points[:,2]))
#         # if sub.maxdist<obj.maxdist:
#         #     return 'infront'
#         # else:
#         #     return 'behind'

        
#     distances = (dleft,dright,dbelow,dabove,dinfront,dbehind)   
#     return RELATIONSHIP_TYPES[np.argmax(distances)]

# def score_relationship(sub : ViewObject,rel_type, obj : ViewObject, output_print=False):
#     assert rel_type in RELATIONSHIP_TYPES
#     sub_bbox, obj_bbox=sub.get_bbox(), obj.get_bbox()

#     #Get the same distances as above
#     dleft= obj_bbox[0] - sub_bbox[2] #TODO: Need these in meters! Fallback: choose explicitly
#     dright= sub_bbox[0] - obj_bbox[2]
#     dbelow= obj.min_z_w - sub.max_z_w
#     dabove= sub.min_z_w - obj.max_z_w
#     dinfront= obj.mindist - sub.maxdist
#     dbehind= sub.mindist - obj.maxdist

#     distances = (dleft,dright,dbelow,dabove,dinfront,dbehind)   

#     if output_print:
#         print(distances)

#     if np.max(distances)>0:
#         score=distances[RELATIONSHIP_TYPES.index(rel_type)] / np.max(distances)
#     else:
#         score= np.max(distances) / distances[RELATIONSHIP_TYPES.index(rel_type)] #Invert if all negative #CARE: correct? smooth?

#     score=distances[RELATIONSHIP_TYPES.index(rel_type)] / np.max(distances)
#     return np.clip(score,0,1)       

'''
Strategy: Weighted center difference, needs z-adjusting for points_c afterall
'''
def get_distances(sub: ViewObject, obj: ViewObject):
    center_difference=obj.center_c - sub.center_c
    center_difference_weighted=center_difference / np.maximum(sub.lengths_c, obj.lengths_c)
    
    dleft= center_difference_weighted[0]
    dright= -dleft
    dbelow= center_difference_weighted[1]
    dabove= -dbelow
    dinfront= center_difference_weighted[2]
    dbehind= -dinfront

    return(dleft,dright,dbelow,dabove,dinfront,dbehind)

def get_relationship_type(sub : ViewObject, obj : ViewObject):
    distances=get_distances(sub, obj)
    return RELATIONSHIP_TYPES[np.argmax(distances)]

def score_relationship_type(sub : ViewObject,rel_type, obj : ViewObject):
    distances=get_distances(sub, obj)
    return np.clip(distances[RELATIONSHIP_TYPES.index(rel_type)] / np.max(distances), 0,1)

def score_color(v: ViewObject, color_name):
    assert color_name in COLOR_NAMES
    color_distances= np.linalg.norm( COLORS-v.color, axis=1 )

    score= np.min(color_distances) / color_distances[COLOR_NAMES.index(color_name)]
    return np.clip(score,0,1)

#Roughly similar to above but for Scene-Graph objects (not view-objects)
#Returns 1.0 if objects are in the same corner, otherwise same logic as above
#CARE: since precise information is lost, scoring can be wrong, e.g. foreground could be left of bottom-left 
CORNER_LOCATION_DICT={ 'top-left':np.array((0.2, 0.2,0.0)) ,'top-right':np.array( (0.8,0.2,0.0)),'bottom-left':np.array( (0.2,0.8,0.0)),'bottom-right':np.array( (0.8,0.8,0.0)),'center':np.array( (0.5,0.5,0.0)), 'foreground':np.array( (0.5,0.5,-0.8)), 'background':np.array( (0.5,0.5,0.8))}
def score_relationship_type_SGO(sub, rel_type, obj):
    if sub.corner==obj.corner: 
        return 1.0

    sub_center=CORNER_LOCATION_DICT[sub.corner]
    obj_center=CORNER_LOCATION_DICT[obj.corner]

    dleft= obj_center[0]-sub_center[0]
    dright= -dleft
    dbelow= sub_center[1]-obj_center[1]
    dabove= -dbelow
    dinfront= obj_center[2]-sub_center[2]
    dbehind= -dinfront

    distances=(dleft,dright,dbelow,dabove,dinfront,dbehind)    
    rel_dist=distances[RELATIONSHIP_TYPES.index(rel_type)]
    if rel_dist>=0.0:
        return 1.0
    else:
        return 0.0

def score_color_SGO(sgo: SceneGraphObject, color_name):
    assert color_name in COLOR_NAMES
    color_distance= np.linalg.norm(COLORS[COLOR_NAMES.index(sgo.color)]-COLORS[COLOR_NAMES.index(color_name)])
    return np.clip(1.0 - color_distance,0,1)    

#Going through all identity-assignment combinations: unfeasible
#All walks: here not possible because Graphs can be (very) incomplete
#Worst-case: evaluate just as before w/o identities?
#-> 2nd one! #Either evaluate w/o identities as before OR eval. relationships separately as before but also check sub.&obj. attribs (disregards identities *between* relationships same/different)

#TODO: score corners? (also see SceneGraphObject)
#CARE: SG should score perfectly to itself, but grounding relations might have different sub/obj
def score_sceneGraph_to_viewObjects_7corners(scene_graph, view_objects):
    #As before: for each relationship: for each possible subject: for each possible object | fingers crossed this is fast enough | no identities | scoring rel-type and color (not corner)    
    MIN_SCORE=0.1 #OPTION: hardest penalty for relationship not found
    best_groundings=[None for i in range(len(scene_graph.relationships))]
    best_scores=[MIN_SCORE for i in range(len(scene_graph.relationships))] 

    if scene_graph.is_empty():
        return 0.0, None

    for i_relation, relation in enumerate(scene_graph.relationships):
        assert type(relation[0] is SceneGraphObject)

        subject_label, rel_type, object_label = relation[0].label, relation[1], relation[2].label
        subject_color, object_color = relation[0].color,relation[2].color
        
        #print(f'Rel {i_relation}: {subject_color} {subject_label} {rel_type} of {object_color} {object_label}')

        for sub in [obj for obj in view_objects if obj.label==subject_label]: 
            for obj in [obj for obj in view_objects if obj.label==object_label]:
                if sub==obj: continue

                #relationship_score= score_relationship(sub, rel_type, obj)
                relationship_score= score_relationship_type(sub, rel_type, obj)
                color_score_sub= score_color(sub, subject_color)
                color_score_obj= score_color(obj, object_color)
                #print(relationship_score, color_score_sub, color_score_obj)

                score=relationship_score*color_score_sub*color_score_obj

                if score>best_scores[i_relation]:
                    best_groundings[i_relation]=(sub,rel_type,obj)
                    best_scores[i_relation]=score

    print("best scores",best_scores)
    return np.prod(best_scores), best_groundings 

'''
As above, but also scores how much the grounded object is the closest one to the subject
-works ✓, scores perfectly to self, now also groundings equal
=> For SG matching
'''
def score_sceneGraph_to_sceneGraph_nnRels(sg0,sg1):
    score0=scenegraph_similarity(sg1,sg0)
    score1=scenegraph_similarity(sg0,sg1)
    return 0.5*(score0+score1)

#Same logic as above, but w/o min-dist scoring 
#TODO: unused_factor
def scenegraph_similarity(source,target):
    MIN_SCORE=0.1 #OPTION: hardest penalty for relationship not found
    #best_groundings=[None for i in range(len(scene_graph.relationships))]
    best_scores=[MIN_SCORE for i in range(len(source.relationships))] 

    if source.is_empty() or target.is_empty():
        return 0.0

    target_objects=extract_scenegraph_objects(target)
    for i_relation, relation in enumerate(source.relationships):
        assert type(relation[0] is SceneGraphObject)

        subject_label, rel_type, object_label = relation[0].label, relation[1], relation[2].label
        subject_color, object_color = relation[0].color,relation[2].color

        for sub in [obj for obj in target_objects if obj.label==subject_label]: 
            for obj in [obj for obj in target_objects if obj.label==object_label]:
                if sub==obj: continue

                relationship_score= score_relationship_type_SGO(sub, rel_type, obj)     
                color_score_sub=score_color_SGO(sub, subject_color)
                color_score_obj=score_color_SGO(obj, object_color)

                score=relationship_score*color_score_sub*color_score_obj

                if score>best_scores[i_relation]:
                    best_scores[i_relation]=score

    return np.prod(best_scores) 

def extract_scenegraph_objects(sg):
    unique_objects=[]
    for candidate in [rel[0] for rel in sg.relationships] + [rel[2] for rel in sg.relationships]:
        obj_already_added=False
        for added_obj in unique_objects:
            if candidate.label==added_obj.label and candidate.color==added_obj.color and candidate.corner==added_obj.corner:
                obj_already_added=True
                break

        if not obj_already_added or True:
            unique_objects.append(candidate)

    return unique_objects        

'''
As above, but also scores how much the grounded object is the closest one to the subject
-works ✓, scores perfectly to self, now also groundings equal
-Tested: unused 0.5, use_nn_score=False best
=> For SG matching
'''
def score_sceneGraph_to_viewObjects_nnRels(scene_graph, view_objects, unused_factor=0.5, use_nn_score=False):
    MIN_SCORE=0.1 #OPTION: hardest penalty for relationship not found
    best_groundings=[None for i in range(len(scene_graph.relationships))]
    best_scores=[MIN_SCORE for i in range(len(scene_graph.relationships))] 

    #Can't score empty graphs 1.0 then apply unused_factor because the factor is not enough to compensate
    if scene_graph.is_empty() or len(view_objects)<2:
        return 0.0, None
    for i_relation, relation in enumerate(scene_graph.relationships):
        assert type(relation[0] is SceneGraphObject)

        subject_label, rel_type, object_label = relation[0].label, relation[1], relation[2].label
        subject_color, object_color = relation[0].color,relation[2].color

        for sub in [obj for obj in view_objects if obj.label==subject_label]: 
            sub_min_dist=np.min( [np.linalg.norm(sub.get_center_c_world() - obj.get_center_c_world()) for obj in view_objects if obj is not sub] )

            for obj in [obj for obj in view_objects if obj.label==object_label]:
                if sub==obj: continue

                relationship_score= score_relationship_type(sub, rel_type, obj)
                color_score_sub= score_color(sub, subject_color)
                color_score_obj= score_color(obj, object_color)
                nn_score= sub_min_dist / np.linalg.norm(sub.get_center_c_world() - obj.get_center_c_world()) #Score whether Obj is Sub's nearest neighbor
                #TODO/CARE: use nn_score?!?!?!

                if use_nn_score:
                    score=relationship_score*color_score_sub*color_score_obj*nn_score
                else:
                    score=relationship_score*color_score_sub*color_score_obj

                if score>best_scores[i_relation]:
                    best_groundings[i_relation]=(sub,rel_type,obj)
                    best_scores[i_relation]=score

    if unused_factor is not None:
        unused_view_objects=[v for v in view_objects]
        for grounding in best_groundings:
            if grounding is not None:
                if grounding[0] in unused_view_objects: unused_view_objects.remove(grounding[0])                    
                if grounding[2] in unused_view_objects: unused_view_objects.remove(grounding[2])

        return np.prod(best_scores) * unused_factor**(len(unused_view_objects)), best_groundings #, unused_view_objects
    else:
        return np.prod(best_scores), best_groundings  



