import pickle
import numpy as np
import cv2
import os
import torch
import sys
#from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models
import torch.nn as nn

from torch_geometric.data import DataLoader #Use the PyG DataLoader

from dataloading.data_loading import Semantic3dDataset, Semantic3dDatasetTriplet
from retrieval import networks
from retrieval.netvlad import NetVLAD, EmbedNet

from semantic.imports import SceneGraph, SceneGraphObject, ViewObject
from semantic.scene_graph_cluster3d_scoring import score_sceneGraph_to_viewObjects_nnRels
from evaluation.utils import evaluate_topK, generate_sanity_check_dataset
import evaluation.utils

def gather_sceneGraph_scores(dataset_train, dataset_test):
    print(f'gather_sceneGraph_scores(): # training: {len(dataset_train)}, # test: {len(dataset_test)}')

    score_dict={} # {test-idx: {train_idx: score} }

    for test_idx in range(len(dataset_test)):
        score_dict[test_idx]={}
        scene_graph=dataset_test.view_scenegraphs[test_idx]
        for train_idx in range(len(dataset_train)):
            score,_=score_sceneGraph_to_viewObjects_nnRels(scene_graph, dataset_train.view_objects[train_idx], unused_factor=0.5)
            score_dict[test_idx][train_idx]=score  

        assert len(score_dict[test_idx])==len(dataset_train)
    assert len(score_dict)==len(dataset_test)

    print('Saving SG-scores...')
    pickle.dump(score_dict, open('scenegraph_scores.pkl','wb'))

def eval_sceneGraph2viewObjects(dataset_train, dataset_test, scenegraph_scores, top_k=(1,3,5,10)):
    assert len(scenegraph_scores)==len(dataset_test)
    assert len(scenegraph_scores[0])==len(dataset_train)

    image_positions_train, image_orientations_train = dataset_train.image_positions, dataset_train.image_orientations
    image_positions_test, image_orientations_test = dataset_test.image_positions, dataset_test.image_orientations
    scene_names_train = dataset_train.image_scene_names
    scene_names_test  = dataset_test.image_scene_names

    retrieval_dict={}

    pos_results  ={k:[] for k in top_k}
    ori_results  ={k:[] for k in top_k}
    scene_results={k:[] for k in top_k}  

    test_indices=np.arange(len(dataset_test))    
    for test_index in test_indices:
        scene_name_gt=scene_names_test[test_index]

        train_indices=np.arange(len(dataset_train))
        scores=np.array([ scenegraph_scores[test_index][train_index] for train_index in train_indices])
        sorted_indices=np.argsort(-1.0*scores) #High->Low

        assert len(sorted_indices)==len(dataset_train)

        pos_dists=np.linalg.norm(image_positions_train[:]-image_positions_test[test_index], axis=1) #CARE: also adds z-distance
        ori_dists=np.abs(image_orientations_train[:]-image_orientations_test[test_index])
        ori_dists=np.minimum(ori_dists, 2*np.pi-ori_dists)

        retrieval_dict[test_index]=sorted_indices[0:np.max(top_k)]

        for k in top_k:
            scene_correct=np.array([scene_name_gt == scene_names_train[retrieved_index] for retrieved_index in sorted_indices[0:k]])
            topk_pos_dists=pos_dists[sorted_indices[0:k]]
            topk_ori_dists=ori_dists[sorted_indices[0:k]]    

            #Append the average pos&ori. errors *for the cases that the scene was hit*
            pos_results[k].append( np.mean( topk_pos_dists[scene_correct==True]) if np.sum(scene_correct)>0 else None )
            ori_results[k].append( np.mean( topk_ori_dists[scene_correct==True]) if np.sum(scene_correct)>0 else None )
            scene_results[k].append( np.mean(scene_correct) ) #Always append the scene-scores
    
    assert len(pos_results[k])==len(ori_results[k])==len(scene_results[k])==len(test_indices)  

    print('Saving retrieval results...')
    pickle.dump(retrieval_dict, open(f'retrievals_SGmatch.pkl','wb'))

    return evaluate_topK(pos_results, ori_results, scene_results) 

#Pure-NetVLAD sanity-check ✓
#Sum-combine sanity check ✓
#Scene-Voting
def eval_netvlad_plus_sceneGraph(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, scenegraph_scores, top_k=(1,3,5,10), combine='sum'):
    assert combine in ('sum','scene-voting->netvlad')
    print(f'eval_netvlad_plus_sceneGraph(): combine {str(combine)}')

    image_positions_train, image_orientations_train = dataset_train.image_positions, dataset_train.image_orientations
    image_positions_test, image_orientations_test = dataset_test.image_positions, dataset_test.image_orientations
    scene_names_train = dataset_train.image_scene_names
    scene_names_test  = dataset_test.image_scene_names

    retrieval_dict={}

    pos_results  ={k:[] for k in top_k}
    ori_results  ={k:[] for k in top_k}
    scene_results={k:[] for k in top_k}    

    test_indices=np.arange(len(dataset_test))    
    for test_index in test_indices:
        scene_name_gt=scene_names_test[test_index]
        train_indices=np.arange(len(dataset_train))

        netvlad_diffs=netvlad_vectors_train-netvlad_vectors_test[test_index]
        netvlad_diffs=np.linalg.norm(netvlad_diffs,axis=1)  

        scores=np.array([ scenegraph_scores[test_index][train_index] for train_index in train_indices]) 

        pos_dists=np.linalg.norm(image_positions_train[:]-image_positions_test[test_index], axis=1) #CARE: also adds z-distance
        ori_dists=np.abs(image_orientations_train[:]-image_orientations_test[test_index])
        ori_dists=np.minimum(ori_dists, 2*np.pi-ori_dists)

        for k in top_k:
            if combine=='sum':
                combined_scores= scores + -1.0*netvlad_diffs 
                sorted_indices=np.argsort( -1.0*combined_scores)[0:k] #High->Low
            if combine=='scene-voting->netvlad':
                indices_netvlad=np.argsort(netvlad_diffs)[0:k] #Low->High
                indices_scenegraph=np.argsort(-1.0*scores)[0:k] #High->Low
                sorted_indices_netvlad,sorted_indices_sg=evaluation.utils.reduceIndices_sceneVoting(scene_names_train, indices_netvlad, indices_scenegraph)
                sorted_indices = sorted_indices_netvlad if len(sorted_indices_netvlad)>0 else sorted_indices_sg # Trust SG-indices if they are united enough to overrule NetVLAD, proved as best approach!

            if k==np.max(top_k): retrieval_dict[test_index]=sorted_indices                

            scene_correct=np.array([scene_name_gt == scene_names_train[retrieved_index] for retrieved_index in sorted_indices])
            if test_index<5 and combine=='scene-voting->netvlad': print(scene_correct)

            topk_pos_dists=pos_dists[sorted_indices]
            topk_ori_dists=ori_dists[sorted_indices]    

            #Append the average pos&ori. errors *for the cases that the scene was hit*
            pos_results[k].append( np.mean( topk_pos_dists[scene_correct==True]) if np.sum(scene_correct)>0 else None )
            ori_results[k].append( np.mean( topk_ori_dists[scene_correct==True]) if np.sum(scene_correct)>0 else None )
            scene_results[k].append( np.mean(scene_correct) ) #Always append the scene-scores
    
    assert len(pos_results[k])==len(ori_results[k])==len(scene_results[k])==len(test_indices)

    print('Saving retrieval results...')
    pickle.dump(retrieval_dict, open('retrievals_netvlad_plus_sceneGraph.pkl','wb'))

    return evaluate_topK(pos_results, ori_results, scene_results)    

if __name__ == "__main__":
    IMAGE_LIMIT=3000
    dataset_train=Semantic3dDataset('data/pointcloud_images_o3d_merged','train',transform=None, image_limit=IMAGE_LIMIT, load_viewObjects=True, load_sceneGraphs=True)
    dataset_test =Semantic3dDataset('data/pointcloud_images_o3d_merged','test', transform=None, image_limit=IMAGE_LIMIT, load_viewObjects=True, load_sceneGraphs=True)    

    if 'gather' in sys.argv:
        gather_sceneGraph_scores(dataset_train, dataset_test)

    scenegraph_scores=pickle.load(open('scenegraph_scores.pkl','rb'))

    if 'SG-match' in sys.argv:
        pos_results, ori_results, scene_results = eval_sceneGraph2viewObjects(dataset_train, dataset_test, scenegraph_scores, top_k=(1,3,5,10))
        print(pos_results, ori_results, scene_results,'\n')        

    if 'NetVLAD+SG-match' in sys.argv:
        netvlad_vectors_train,netvlad_vectors_test=pickle.load(open('netvlad_vectors.pkl','rb'))
        pos_results, ori_results, scene_results = eval_netvlad_plus_sceneGraph(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, scenegraph_scores, top_k=(1,3,5,10), combine='sum')
        print(pos_results, ori_results, scene_results,'\n')        
        pos_results, ori_results, scene_results = eval_netvlad_plus_sceneGraph(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, scenegraph_scores, top_k=(1,3,5,10), combine='scene-voting->netvlad')
        print(pos_results, ori_results, scene_results,'\n')        

