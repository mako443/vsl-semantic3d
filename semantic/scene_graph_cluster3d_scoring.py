import numpy as np
from graphics.imports import CLASSES_DICT, CLASSES_COLORS, IMAGE_WIDHT, IMAGE_HEIGHT
from .imports import ViewObject, COLORS, COLOR_NAMES, CORNERS, CORNER_NAMES, RELATIONSHIP_TYPES


#TODO: how to resolve the 'terrain angled in front of hard-scape' situation?
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

    #With too much overlap, it can only be in front or behind
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
