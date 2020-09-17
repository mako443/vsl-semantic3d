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

from semantic.imports import SceneGraph, SceneGraphObject, ViewObject
from semantic.scene_graph_cluster3d_scoring import score_sceneGraph_to_viewObjects_nnRels
from evaluation.utils import evaluate_topK, generate_sanity_check_dataset

from visual_semantic.visual_semantic_embedding import VisualSemanticEmbedding
from geometric.visual_graph_embedding import create_image_model_vgg11, VisualGraphEmbeddingNetVLAD

'''
Evaluation between a NetVLAD-model and a model that predicts the NetVLAD vector cross-modal
distance-sum: predicts feature vectors separately, sums up the distances (both sides multi-modal)
mean-vector-test: uses the mean vector on the test-side, regular NetVLAD vector on the train side (query multi-modal)
mean-vector: uses the mean vectors on both sides (both sides multi-modal)
cross-modal: uses the predicted vectors on the test side, regular NetVLAD vectors on the train side (one modality per side)
embed-only: uses only the graph embeddings (both sides semantic-only, but "anchored" on NetVLAD)

-> Distance-Sum & Mean-Vector work the same and perform equal to pure-NV, mean-vector-test slightly worse, cross-modal bad, embed-only bad
-> VGE-NV does not precisely predict the NetVLAD vectors...
'''
def evaluate_netvlad_predictions(dataset_train, dataset_test, netvlad_train, netvlad_test, embedding_train, embedding_test ,top_k=(1,3,5,10), combine='distance-sum'):
    assert combine in ('distance-sum','mean-vector-test','mean-vector','cross-modal','embed-only')
    print('\n evaluate_netvlad_predictions():', combine)

    image_positions_train, image_orientations_train = dataset_train.image_positions, dataset_train.image_orientations
    image_positions_test, image_orientations_test = dataset_test.image_positions, dataset_test.image_orientations
    scene_names_train = dataset_train.image_scene_names
    scene_names_test  = dataset_test.image_scene_names    

    retrieval_dict={}

    pos_results  ={k:[] for k in top_k}
    ori_results  ={k:[] for k in top_k}
    scene_results={k:[] for k in top_k}   

    mean_vectors_train=0.5*(netvlad_train+embedding_train)
    mean_vectors_test =0.5*(netvlad_test +embedding_test )

    check_indices=np.arange(len(dataset_test))
    for idx in check_indices:
        scene_name_gt=scene_names_test[idx]

        if combine=='distance-sum':
            netvlad_distances  = np.linalg.norm( netvlad_train[:]  - netvlad_test[idx] , axis=1 )
            embedding_distances= np.linalg.norm( embedding_train[:]-embedding_test[idx], axis=1 )
            combined_distances=netvlad_distances+embedding_distances
            sorted_indices=np.argsort(combined_distances)
        if combine=='mean-vector-test':
            combined_distances= np.linalg.norm( netvlad_train[:] - mean_vectors_test[idx], axis=1)
            sorted_indices=np.argsort(combined_distances)
        if combine=='mean-vector':
            combined_distances= np.linalg.norm( mean_vectors_train[:] - mean_vectors_test[idx], axis=1)
            sorted_indices=np.argsort(combined_distances)
        if combine=='cross-modal':
            combined_distances= np.linalg.norm( netvlad_train[:] - embedding_test[idx], axis=1)
            sorted_indices=np.argsort(combined_distances)
        if combine=='embed-only':
            combined_distances=np.linalg.norm( embedding_train[:] - embedding_test[idx], axis=1)
            sorted_indices=np.argsort(combined_distances)

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
    pickle.dump(retrieval_dict, open(f'retrievals_VGE_NV_{combine}.pkl','wb'))

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

    dataset_train=Semantic3dDataset('data/pointcloud_images_o3d_merged','train',transform=transform, image_limit=IMAGE_LIMIT, load_viewObjects=True, load_sceneGraphs=True, return_graph_data=True)
    dataset_test =Semantic3dDataset('data/pointcloud_images_o3d_merged','test', transform=transform, image_limit=IMAGE_LIMIT, load_viewObjects=True, load_sceneGraphs=True, return_graph_data=True)

    dataloader_train=DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False) #CARE: put shuffle off
    dataloader_test =DataLoader(dataset_test , batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False)        

    #TODO: re-do with VGE(base version, not NV)

    #Create the models
    image_encoder=networks.get_encoder_resnet18()
    image_encoder.requires_grad_(False) #Don't train encoder
    netvlad_layer=NetVLAD(num_clusters=NUM_CLUSTERS, dim=512, alpha=ALPHA)
    netvlad_model=EmbedNet(image_encoder, netvlad_layer)

    netvlad_model_name='model_netvlad_l3000_b6_g0.75_c8_a10.0.pth'
    print('Model:',netvlad_model_name)
    netvlad_model.load_state_dict(torch.load('models/'+netvlad_model_name))
    netvlad_model.eval()
    netvlad_model.cuda()

    visual_graph_embedding=VisualGraphEmbeddingNetVLAD(netvlad_model, EMBED_DIM)
    visual_graph_embedding_model_name='model_vgeNV_l3000_b6_g0.75_e300_sTrue_m1.0_d0.001.pth'
    print('Model:',visual_graph_embedding_model_name)
    visual_graph_embedding.load_state_dict(torch.load('models/'+visual_graph_embedding_model_name))
    visual_graph_embedding.eval()
    visual_graph_embedding.cuda()

    #Gather all features
    print('Building NetVLAD vectors')
    netvlad_vectors_train, netvlad_vectors_test=torch.tensor([]).cuda(), torch.tensor([]).cuda()
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader_test):
            a=batch['images']
            a_out=netvlad_model(a.cuda())
            netvlad_vectors_test=torch.cat((netvlad_vectors_test,a_out))
        for i_batch, batch in enumerate(dataloader_train):
            a=batch['images']
            a_out=netvlad_model(a.cuda())
            netvlad_vectors_train=torch.cat((netvlad_vectors_train,a_out))   
    netvlad_vectors_train=netvlad_vectors_train.cpu().detach().numpy()
    netvlad_vectors_test=netvlad_vectors_test.cpu().detach().numpy()    

    print('Building Embedding vectors')
    embedding_vectors_train, embedding_vectors_test=torch.tensor([]).cuda(), torch.tensor([]).cuda()
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader_test):
            a=batch['graphs']
            a_out=visual_graph_embedding.encode_graphs(a.to('cuda'))
            embedding_vectors_test=torch.cat((embedding_vectors_test,a_out))
        for i_batch, batch in enumerate(dataloader_train):
            a=batch['graphs']
            a_out=visual_graph_embedding.encode_graphs(a.to('cuda'))
            embedding_vectors_train=torch.cat((embedding_vectors_train,a_out))  
    embedding_vectors_train=embedding_vectors_train.cpu().detach().numpy()
    embedding_vectors_test=embedding_vectors_test.cpu().detach().numpy()                 

    assert netvlad_vectors_test.shape==embedding_vectors_test.shape and netvlad_vectors_train.shape==embedding_vectors_train.shape
    # d={'netvlad_train':netvlad_vectors_train, 'netvlad_test':netvlad_vectors_test, 'embed_train':embedding_vectors_train, 'embed_test':embedding_vectors_test}
    # pickle.dump(d, open('vge_vectors.pkl','wb'))
    # print('vecs written')
    # quit()

    # print("Train:",np.linalg.norm(netvlad_vectors_train-embedding_vectors_train)/np.linalg.norm(netvlad_vectors_train))
    # print("Test:", np.linalg.norm(netvlad_vectors_test -embedding_vectors_test )/np.linalg.norm(netvlad_vectors_test ))
    # quit()

    # pos_results, ori_results, scene_results=evaluate_netvlad_predictions(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, embedding_vectors_train, embedding_vectors_test, combine='embed-only')
    # print(pos_results, ori_results, scene_results)  
    # quit()    

    #Run evaluation
    pos_results, ori_results, scene_results=evaluate_netvlad_predictions(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, embedding_vectors_train, embedding_vectors_test, combine='distance-sum')
    print(pos_results, ori_results, scene_results)  

