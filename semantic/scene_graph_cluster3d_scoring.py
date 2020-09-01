import numpy as np
from graphics.imports import CLASSES_DICT, CLASSES_COLORS, IMAGE_WIDHT, IMAGE_HEIGHT
from .imports import SceneGraph,SceneGraphObject, ViewObject, COLORS, COLOR_NAMES, CORNERS, CORNER_NAMES, RELATIONSHIP_TYPES

#TODO: how to resolve the 'terrain angled in front of hard-scape' situation?
#Select the relationship type as the direction with the largest distance
#CARE: Comparing in different Units, pixels vs. world-coordinates!!
def get_relationship_type(sub : ViewObject, obj : ViewObject):
    sub_bbox, obj_bbox=sub.get_bbox(), obj.get_bbox()
    #Compare left/right in image-coords
    dleft= obj_bbox[0] - sub_bbox[2]
    dright= sub_bbox[0] - obj_bbox[2]
    #Compare below/above in world-coords
    dbelow= obj.min_z_w - sub.max_z_w
    dabove= sub.min_z_w - obj.max_z_w
    dinfront= obj.mindist - sub.maxdist
    dbehind= sub.mindist - obj.maxdist

    #With too much overlap, it can only be in front or behind #CARE: this is not reversible
    if np.sum(( dleft<0, dright<0, dbelow<0, dabove<0 ))>=3:
        dleft, dright=-1e6,-1e6
        dbelow, dabove=-1e6,-1e6
        # #sub_center_z=1/2*(np.min(sub.points[:,2])+ np.max(sub.points[:,2]))
        # #obj_center_z=1/2*(np.min(obj.points[:,2])+ np.max(obj.points[:,2]))
        # if sub.maxdist<obj.maxdist:
        #     return 'infront'
        # else:
        #     return 'behind'

        
    distances = (dleft,dright,dbelow,dabove,dinfront,dbehind)   
    return RELATIONSHIP_TYPES[np.argmax(distances)] 

def score_relationship(sub : ViewObject,rel_type, obj : ViewObject, output_print=False):
    assert rel_type in RELATIONSHIP_TYPES
    sub_bbox, obj_bbox=sub.get_bbox(), obj.get_bbox()

    #Get the same distances as above
    dleft= obj_bbox[0] - sub_bbox[2] #TODO: Need these in meters! Fallback: choose explicitly
    dright= sub_bbox[0] - obj_bbox[2]
    dbelow= obj.min_z_w - sub.max_z_w
    dabove= sub.min_z_w - obj.max_z_w
    dinfront= obj.mindist - sub.maxdist
    dbehind= sub.mindist - obj.maxdist

    distances = (dleft,dright,dbelow,dabove,dinfront,dbehind)   

    if output_print:
        print(distances)

    if np.max(distances)>0:
        score=distances[RELATIONSHIP_TYPES.index(rel_type)] / np.max(distances)
    else:
        score= np.max(distances) / distances[RELATIONSHIP_TYPES.index(rel_type)] #Invert if all negative #CARE: correct? smooth?

    score=distances[RELATIONSHIP_TYPES.index(rel_type)] / np.max(distances)
    return np.clip(score,0,1)       

def score_color(v: ViewObject, color_name):
    assert color_name in COLOR_NAMES
    color_distances= np.linalg.norm( COLORS-v.color, axis=1 )

    score= np.min(color_distances) / color_distances[COLOR_NAMES.index(color_name)]
    return np.clip(score,0,1)

#Going through all identity-assignment combinations: unfeasible
#All walks: here not possible because Graphs can be (very) incomplete
#Worst-case: evaluate just as before w/o identities?
#-> 2nd one! #Either evaluate w/o identities as before OR eval. relationships separately as before but also check sub.&obj. attribs (disregards identities *between* relationships same/different)
def score_sceneGraph_to_viewObjects(scene_graph, view_objects):
    #As before: for each relationship: for each possible subject: for each possible object | fingers crossed this is fast enough | no identities | scoring rel-type and color (not corner)    
    MIN_SCORE=0.1 #OPTION: hardest penalty for relationship not found
    best_groundings=[None for i in range(len(scene_graph.relationships))]
    best_scores=[MIN_SCORE for i in range(len(scene_graph.relationships))] 

    for i_relation, relation in enumerate(scene_graph.relationships):
        assert type(relation[0] is SceneGraphObject)

        subject_label, rel_type, object_label = relation[0].label, relation[1], relation[2].label
        subject_color, object_color = relation[0].color,relation[0].color

        for sub in [obj for obj in view_objects if obj.label==subject_label]: 
            for obj in [obj for obj in view_objects if obj.label==object_label]:
                if sub==obj: continue

                relationship_score= score_relationship(sub, rel_type, obj)
                color_score_sub= score_color(sub, subject_color)
                color_score_obj= score_color(obj, object_color)

                score=relationship_score*color_score_sub*color_score_obj

                if score>best_scores[i_relation]:
                    best_groundings[i_relation]=(sub,rel_type,obj)
                    best_scores[i_relation]=score

    print("best scores",best_scores)
    return np.prod(best_scores), best_groundings    


