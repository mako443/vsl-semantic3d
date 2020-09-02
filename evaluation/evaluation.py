import pickle
import numpy as np
import cv2
import os
from dataloading.data_loading import Semantic3dDataset
#from semantic.scene_graph import ground_scenegraph_to_patches, Relationship2
#from semantic.patches import Patch
from semantic.imports import SceneGraph, SceneGraphObject, ViewObject
from semantic.scene_graph_cluster3d_scoring import score_sceneGraph_to_viewObjects
from evaluation.utils import evaluate_topK


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
-Handle empty Scene Graphs (0.0 score ✓)
'''
def scenegraph_to_viewObjects(base_path, top_k=(1,5,10)):
    CHECK_COUNT=50

    dataset=Semantic3dDataset(base_path)

    pos_results  ={k:[] for k in top_k}
    ori_results  ={k:[] for k in top_k}
    scene_results={k:[] for k in top_k}

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
                distance_sum[k]   +=np.mean( topk_loc_dists[scene_correct==True] ) #ERROR! 0 mean added if no scene hit
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

def netvlad_retrieval(data_loader_train, data_loader_test, model, top_k=(1,5,10), random_features=False):
    CHECK_COUNT=50
    print(f'# training: {len(data_loader_train.dataset)}, # test: {len(data_loader_test.dataset)}')

    if random_features:
        print('Using random vectors (sanity check)')
        netvlad_vectors_train=np.random.rand(len(data_loader_train.dataset),2)
        netvlad_vectors_test=np.random.rand(len(data_loader_test.dataset),2)        
    else:
        print('Building NetVLAD vectors...')
        netvlad_vectors_train, netvlad_vectors_test=torch.tensor([]).cuda(), torch.tensor([]).cuda()

        with torch.no_grad():
            for i_batch, batch in enumerate(data_loader_test): #loading p,n somewhat redundant
                a,p,n=batch
                a_out=model(a.cuda())
                netvlad_vectors_test=torch.cat((netvlad_vectors_test,a_out))
            for i_batch, batch in enumerate(data_loader_train):
                a,p,n=batch
                a_out=model(a.cuda())
                netvlad_vectors_train=torch.cat((netvlad_vectors_train,a_out))        

        netvlad_vectors_train=netvlad_vectors_train.cpu().detach().numpy()
        netvlad_vectors_test=netvlad_vectors_test.cpu().detach().numpy()

    # image_positions_train, image_orientations_train = data_loader_train.dataset.image_positions, data_loader_train.dataset.image_orientations
    # image_positions_test, image_orientations_test = data_loader_test.dataset.image_positions, data_loader_test.dataset.image_orientations

    pos_results  ={k:[] for k in top_k}
    ori_results  ={k:[] for k in top_k}
    scene_results={k:[] for k in top_k}

    print('evaluating random indices...')
    for i in range(CHECK_COUNT):
        idx=np.random.choice(range(len(netvlad_vectors_test)))
        scene_name_gt=data_loader_test.dataset.get_scene_name(idx)

        netvlad_diffs=netvlad_vectors_train-netvlad_vectors_test[idx]
        netvlad_diffs=np.linalg.norm(netvlad_diffs,axis=1)   

        sorted_indices=np.argsort(netvlad_diffs) 
        pos_dists=np.linalg.norm(image_positions_train[:]-image_positions_test[idx], axis=1)
        ori_dists=np.abs(image_orientations_train[:]-image_orientations_test[idx])
        ori_dists=np.minimum(ori_dists, 2*np.pi-ori_dists)

        for k in top_k:
            scene_correct=np.array([scene_name_gt == data_loader_train.dataset.get_scene_name(retrieved_index) for retrieved_index in sorted_indices[0:k]])
            topk_pos_dists=pos_dists[sorted_indices[0:k]]
            topk_ori_dists=ori_dists[sorted_indices[0:k]]    

            #Append the average pos&ori. errors *for the cases that the scene was hit*
            pos_results[k].append( np.mean( topk_pos_dists[scene_correct==True]) if np.sum(scene_correct)>0 else None )
            ori_results[k].append( np.mean( topk_ori_dists[scene_correct==True]) if np.sum(scene_correct)>0 else None )
            scene_results[k].append( np.mean(scene_correct) ) #Always append the scene-scores
    
    assert len(pos_results[k])==len(ori_results[k])==len(scene_results[k])==CHECK_COUNT

    return evaluate_topK(pos_results, ori_results, scene_results)


if __name__ == "__main__":
    quit()
    distance_avg, orientation_avg, scene_avg=scenegraph_to_viewObjects('data/pointcloud_images_o3d')
    print(distance_avg, orientation_avg, scene_avg)
