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
from evaluation.utils import evaluate_topK, generate_sanity_check_dataset
from geometric.graph_embedding import GraphEmbedding

def eval_graph2graph(dataset_train, dataset_test, embedding_train, embedding_test ,top_k=(1,3,5,10), similarity='l2'):
    assert similarity in ('l2',)
    print('\n evaluate_netvlad_predictions():', similarity)

    image_positions_train, image_orientations_train = dataset_train.image_positions, dataset_train.image_orientations
    image_positions_test, image_orientations_test = dataset_test.image_positions, dataset_test.image_orientations
    scene_names_train = dataset_train.image_scene_names
    scene_names_test  = dataset_test.image_scene_names    

    retrieval_dict={}

    pos_results  ={k:[] for k in top_k}
    ori_results  ={k:[] for k in top_k}
    scene_results={k:[] for k in top_k}   

    check_indices=np.arange(len(dataset_test))
    for idx in check_indices:
        scene_name_gt=scene_names_test[idx]

        #CARE: Make sure to use the correct similarity
        if similarity=='l2': #L2-difference, e.g. from TipletMarginLoss
            vector_diffs=embedding_train-embedding_test[idx]
            vector_diffs=np.linalg.norm(vector_diffs,axis=1) 
            sorted_indices=np.argsort(vector_diffs)     #Low->High differences

        assert len(sorted_indices)==len(dataset_train)

        pos_dists=np.linalg.norm(image_positions_train[:]-image_positions_test[idx], axis=1) #CARE: also adds z-distance
        ori_dists=np.abs(image_orientations_train[:]-image_orientations_test[idx])
        ori_dists=np.minimum(ori_dists, 2*np.pi-ori_dists)

        retrieval_dict[idx]=sorted_indices[0:np.max(top_k)]

        for k in top_k:
            scene_correct=np.array([scene_name_gt == scene_names_train[retrieved_index] for retrieved_index in sorted_indices[0:k]])
            topk_pos_dists=pos_dists[sorted_indices[0:k]]
            topk_ori_dists=ori_dists[sorted_indices[0:k]]    

            #Append the average pos&ori. errors *for the cases that the scene was hit*
            pos_results[k].append( np.mean( topk_pos_dists[scene_correct==True]) if np.sum(scene_correct)>0 else None )
            ori_results[k].append( np.mean( topk_ori_dists[scene_correct==True]) if np.sum(scene_correct)>0 else None )
            scene_results[k].append( np.mean(scene_correct) ) #Always append the scene-scores
    
    assert len(pos_results[k])==len(ori_results[k])==len(scene_results[k])==len(check_indices)  

    print('Saving retrieval results...')
    pickle.dump(retrieval_dict, open(f'retrievals_PureGE_{similarity}.pkl','wb'))

    return evaluate_topK(pos_results, ori_results, scene_results)  

'''
Module for pure Geometric (Graph->Graph) retrieval
'''

if __name__ == "__main__":
    IMAGE_LIMIT=3000
    BATCH_SIZE=6
    NUM_CLUSTERS=8
    #EMBED_DIM=300
    ALPHA=10.0
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])     
    dataset_train=Semantic3dDataset('data/pointcloud_images_o3d_merged','train',transform=transform, image_limit=IMAGE_LIMIT, load_viewObjects=True, load_sceneGraphs=True, return_graph_data=True)
    dataset_test =Semantic3dDataset('data/pointcloud_images_o3d_merged','test', transform=transform, image_limit=IMAGE_LIMIT, load_viewObjects=True, load_sceneGraphs=True, return_graph_data=True)
    dataloader_train=DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False) #CARE: put shuffle off
    dataloader_test =DataLoader(dataset_test , batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False)         

    print('## Evaluation: Pure GE scoring')
    EMBED_DIM=100 #CARE: Make sure this matches
    model=GraphEmbedding(embedding_dim=EMBED_DIM)
    model_name='model_GraphEmbed_l3000_b12_g0.75_e100_sTrue_m1.0.pth'
    model.load_state_dict(torch.load('models/'+model_name))
    model.eval()
    model.cuda()
    print('Model:',model_name)

    print('Building Embedding vectors')
    embedding_vectors_train, embedding_vectors_test=torch.tensor([]).cuda(), torch.tensor([]).cuda()
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader_test):
            a=batch['graphs']
            a_out=model(a.to('cuda'))
            embedding_vectors_test=torch.cat((embedding_vectors_test,a_out))
        for i_batch, batch in enumerate(dataloader_train):
            a=batch['graphs']
            a_out=model(a.to('cuda'))
            embedding_vectors_train=torch.cat((embedding_vectors_train,a_out))     
    embedding_vectors_train=embedding_vectors_train.cpu().detach().numpy()
    embedding_vectors_test=embedding_vectors_test.cpu().detach().numpy()            

    pos_results, ori_results, scene_results=eval_graph2graph(dataset_train, dataset_test, embedding_vectors_train, embedding_vectors_test, similarity='l2')
    print(pos_results, ori_results, scene_results)  