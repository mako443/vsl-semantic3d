import pickle
import numpy as np
import cv2
import os
from dataloading.data_loading import Semantic3dDataset
from semantic.scene_graph import ground_scenegraph_to_patches, Relationship2
from semantic.patches import Patch

'''
Module for evaluation
All results as {k: avg-distance-err, }, {k: avg-orientation-err }, {k: avg-scene-hit, } | distance&ori.-errors are reported among scene-hits
'''

def scenegraph_to_patches(base_path, top_k=(1,5,10)):
    check_count=50

    dataset=Semantic3dDataset(base_path)

    distance_sum={ k:0 for k in top_k }
    orientation_sum={ k:0 for k in top_k }
    scene_sum={ k:0 for k in top_k }    

    for check_idx in range(check_count):
        #Score SG vs. all images
        grounding_scores=np.zeros(len(dataset))
        for i in range(len(dataset)):
            score,_ = ground_scenegraph_to_patches( dataset.image_scenegraphs[i], dataset.image_patches[i] )
            grounding_scores[i]=score
        grounding_scores[check_idx]=0.0 #Don't score vs. self

        sorted_indices=np.argsort( -1.0*grounding_scores) #Sort highest -> lowest scores

        location_dists=dataset.image_poses[:,0:3]-dataset.image_poses[check_idx,0:3]
        location_dists=np.linalg.norm(location_dists,axis=1)      

        orientation_dists=np.abs(dataset.image_poses[:,3]-dataset.image_poses[check_idx,3]) 
        orientation_dists=np.minimum(orientation_dists,2*np.pi-orientation_dists)          

        scene_name_gt=dataset.get_scene_name(check_idx)

        for k in top_k:
            scene_correct= np.array([scene_name_gt == dataset.get_scene_name(retrieved_index) for retrieved_index in sorted_indices[0:k] ])
            topk_loc_dists=location_dists[sorted_indices[0:k]]
            topk_ori_dists=orientation_dists[sorted_indices[0:k]]

            if np.sum(scene_correct)>0:
                distance_sum[k]   +=np.mean( topk_loc_dists[scene_correct==True] )
                orientation_sum[k]+=np.mean( topk_ori_dists[scene_correct==True] )
                scene_sum[k]      +=np.mean(scene_correct)            

    distance_avg, orientation_avg, scene_avg={},{},{}
    for k in top_k:
        distance_avg[k] = distance_sum[k]/check_count
        orientation_avg[k] = orientation_sum[k]/check_count
        scene_avg[k]= scene_sum[k]/check_count
        distance_avg[k],orientation_avg[k], scene_avg[k]=np.float16(distance_avg[k]),np.float16(orientation_avg[k]),np.float16(scene_avg[k]) #Make numbers more readable    

    return distance_avg, orientation_avg, scene_avg


if __name__ == "__main__":
    distance_avg, orientation_avg, scene_avg=scenegraph_to_patches('data/pointcloud_images_3_2_depth')
    print(distance_avg, orientation_avg, scene_avg)
