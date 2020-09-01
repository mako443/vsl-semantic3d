import pickle
import numpy as np
import cv2
import os
from dataloading.data_loading import Semantic3dDataset
#from semantic.scene_graph import ground_scenegraph_to_patches, Relationship2
#from semantic.patches import Patch
from semantic.imports import SceneGraph, SceneGraphObject, ViewObject
from semantic.scene_graph_cluster3d_scoring import score_sceneGraph_to_viewObjects


'''
Module for evaluation
All results as {k: avg-distance-err, }, {k: avg-orientation-err }, {k: avg-scene-hit, } | distance&ori.-errors are reported among scene-hits
'''

'''
-TODO: general & cleaner function to prepare results
'''

#DEPRECATED
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

'''
Matching SGs analytically to the View-Objects from 3D-Clustering

4 scenes, random                                : {1: 1.826, 5: 8.55, 10: 12.336} {1: 0.3015, 5: 1.085, 10: 1.426} {1: 0.2, 5: 0.248, 10: 0.26} CARE:Increasing because of more scene-hits?
4 scenes, scenegraph_for_view_cluster3d_7corners: {1: 8.52, 5: 10.16, 10: 10.72} {1: 0.867, 5: 1.057, 10: 1.1} {1: 0.6, 5: 0.532, 10: 0.524}

-simple check close-by / far away
-check top-hits
'''
def scenegraph_to_viewObjects(base_path, top_k=(1,5,10)):
    CHECK_COUNT=50

    dataset=Semantic3dDataset(base_path)

    distance_sum={ k:0 for k in top_k }
    orientation_sum={ k:0 for k in top_k }
    scene_sum={ k:0 for k in top_k } 

    for i_check_idx,check_idx in enumerate(np.random.randint(len(dataset), size=CHECK_COUNT)):
        print(f'\r index {i_check_idx} of {CHECK_COUNT}', end='')
        #Score SG vs. all images
        scene_graph=dataset.view_scenegraphs[check_idx]
        scores=np.zeros(len(dataset))
        for i in range(len(dataset)):
            score,_=score_sceneGraph_to_viewObjects(scene_graph, dataset.view_objects[i])
            scores[i]=score
        #scores=np.random.rand(len(dataset))
        scores[check_idx]=0.0 #Don't score vs. self
        
        sorted_indices=np.argsort(-1.0*scores) #Sort highest -> lowest scores

        location_dists=dataset.image_positions[:,:]-dataset.image_positions[check_idx,:]
        location_dists=np.linalg.norm(location_dists,axis=1)    

        orientation_dists=np.abs(dataset.image_orientations[:]-dataset.image_orientations[check_idx]) 
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
    print()

    distance_avg, orientation_avg, scene_avg={},{},{}
    for k in top_k:
        distance_avg[k] = distance_sum[k]/CHECK_COUNT
        orientation_avg[k] = orientation_sum[k]/CHECK_COUNT
        scene_avg[k]= scene_sum[k]/CHECK_COUNT
        distance_avg[k],orientation_avg[k], scene_avg[k]=np.float16(distance_avg[k]),np.float16(orientation_avg[k]),np.float16(scene_avg[k]) #Make numbers more readable    

    return distance_avg, orientation_avg, scene_avg        


if __name__ == "__main__":
    distance_avg, orientation_avg, scene_avg=scenegraph_to_viewObjects('data/pointcloud_images_o3d')
    print(distance_avg, orientation_avg, scene_avg)
