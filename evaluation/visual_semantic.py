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

from dataloading.data_loading import Semantic3dDataset
from retrieval import networks
from retrieval.netvlad import NetVLAD, EmbedNet

from evaluation.utils import evaluate_topK, generate_sanity_check_dataset
import evaluation.utils

from visual_semantic.visual_semantic_embedding import VisualSemanticEmbedding
from visual_semantic.semantic_embedding import SemanticEmbedding

def gather_SE_vectors(dataloader_train, dataloader_test, model):
    #Gather all features
    print('Building SE vectors')
    embed_vectors_train, embed_vectors_test=torch.tensor([]).cuda(), torch.tensor([]).cuda()
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader_train):
            a=batch['captions']
            a_out=model(a)
            embed_vectors_train=torch.cat((embed_vectors_train,a_out))   
        for i_batch, batch in enumerate(dataloader_test):
            a=batch['captions']
            a_out=model(a)
            embed_vectors_test=torch.cat((embed_vectors_test,a_out))
    embed_vectors_train=embed_vectors_train.cpu().detach().numpy()
    embed_vectors_test=embed_vectors_test.cpu().detach().numpy() 
    embed_dim=embed_vectors_test.shape[1]

    pickle.dump((embed_vectors_train, embed_vectors_test), open(f'features_SE_e{embed_dim}.pkl','wb'))
    print('Saved SE-vectors')

'''
Goes from query-side embed-vectors to db-side embed vectors
Used for vectors from SE, VSE-UE and VSE-NV
'''
def eval_SE_scoring(dataset_train, dataset_test, embedding_train, embedding_test ,top_k=(1,3,5,10), reduce_indices=None):
    assert len(embedding_train)==len(dataset_train) and len(embedding_test)==len(dataset_test)
    print(f'eval_SE_scoring(): # training: {len(dataset_train)}, # test: {len(dataset_test)}')
    print('Reduce indices:',reduce_indices)    

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

        #Score cosine similarity
        scores=embedding_train@embedding_test[test_index]
        
        assert len(scores)==len(dataset_train)

        pos_dists=np.linalg.norm(image_positions_train[:]-image_positions_test[test_index], axis=1) #CARE: also adds z-distance
        ori_dists=np.abs(image_orientations_train[:]-image_orientations_test[test_index])
        ori_dists=np.minimum(ori_dists, 2*np.pi-ori_dists)

        #retrieval_dict[test_index]=sorted_indices[0:np.max(top_k)]

        for k in top_k:
            if reduce_indices is None:
                sorted_indices=np.argsort(-1.0*scores)[0:k] #High->Low
            if reduce_indices=='scene-voting':
                sorted_indices=np.argsort(-1.0*scores)[0:k] #High->Low
                sorted_indices=evaluation.utils.reduceIndices_sceneVoting(scene_names_train, sorted_indices)

            if k==np.max(top_k): retrieval_dict[test_index]=sorted_indices

            scene_correct=np.array([scene_name_gt == scene_names_train[retrieved_index] for retrieved_index in sorted_indices[0:k]])
            topk_pos_dists=pos_dists[sorted_indices[0:k]]
            topk_ori_dists=ori_dists[sorted_indices[0:k]]    

            #Append the average pos&ori. errors *for the cases that the scene was hit*
            pos_results[k].append( np.mean( topk_pos_dists[scene_correct==True]) if np.sum(scene_correct)>0 else None )
            ori_results[k].append( np.mean( topk_ori_dists[scene_correct==True]) if np.sum(scene_correct)>0 else None )
            scene_results[k].append( np.mean(scene_correct) ) #Always append the scene-scores
    
    assert len(pos_results[k])==len(ori_results[k])==len(scene_results[k])==len(test_indices)  

    print('Saving retrieval results...')
    pickle.dump(retrieval_dict, open(f'retrievals_SE.pkl','wb'))

    return evaluate_topK(pos_results, ori_results, scene_results)      

if __name__ == "__main__":
    IMAGE_LIMIT=3000
    BATCH_SIZE=6
    NUM_CLUSTERS=8
    EMBED_DIM=300
    ALPHA=10.0

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])  

    dataset_train=Semantic3dDataset('data/pointcloud_images_o3d_merged','train',transform=transform, image_limit=IMAGE_LIMIT, load_viewObjects=True, load_sceneGraphs=True, return_captions=True)
    dataset_test =Semantic3dDataset('data/pointcloud_images_o3d_merged','test', transform=transform, image_limit=IMAGE_LIMIT, load_viewObjects=True, load_sceneGraphs=True, return_captions=True)

    dataloader_train=DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False) #CARE: put shuffle off
    dataloader_test =DataLoader(dataset_test , batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False)        

    if 'gather' in sys.argv:
        #Gather SE
        EMBED_DIM_SEMANTIC=100
        semantic_embedding=SemanticEmbedding(dataset_train.get_known_words(),EMBED_DIM_SEMANTIC)
        semantic_embedding_model_name='model_SemanticEmbed_l3000_b12_g0.75_e100_sTrue_m1.0_lr0.005.pth'
        print('Model:',semantic_embedding_model_name)
        semantic_embedding.load_state_dict(torch.load('models/'+semantic_embedding_model_name))
        semantic_embedding.eval()
        semantic_embedding.cuda()         
        gather_SE_vectors(dataloader_train, dataloader_test, semantic_embedding)

    if 'SE-match' in sys.argv:
        se_vectors_filename='features_SE_e100.pkl'
        se_vectors_train, se_vectors_test=pickle.load(open('evaluation_res/'+se_vectors_filename,'rb')); print('Using vectors',se_vectors_filename)
        pos_results, ori_results, scene_results=eval_SE_scoring(dataset_train, dataset_test, se_vectors_train, se_vectors_test ,top_k=(1,3,5,10), reduce_indices=None)
        print(pos_results, ori_results, scene_results,'\n') 
        pos_results, ori_results, scene_results=eval_SE_scoring(dataset_train, dataset_test, se_vectors_train, se_vectors_test ,top_k=(1,3,5,10), reduce_indices='scene-voting')
        print(pos_results, ori_results, scene_results,'\n')         

        # se_vectors_filename='features_SE_e1024.pkl'
        # se_vectors_train, se_vectors_test=pickle.load(open('evaluation_res/'+se_vectors_filename,'rb')); print('Using vectors',se_vectors_filename)
        # pos_results, ori_results, scene_results=eval_SE_scoring(dataset_train, dataset_test, se_vectors_train, se_vectors_test ,top_k=(1,3,5,10))
        # print(pos_results, ori_results, scene_results,'\n')            