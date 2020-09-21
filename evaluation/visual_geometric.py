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
import evaluation.utils

from visual_semantic.visual_semantic_embedding import VisualSemanticEmbedding

from geometric.graph_embedding import GraphEmbedding
from geometric.visual_graph_embedding import create_image_model_vgg11, VisualGraphEmbeddingNetVLAD, VisualGraphEmbedding

def gather_GE_vectors(dataloader_train, dataloader_test, model):
    #Gather all features
    print('Building GE vectors')
    embed_vectors_train, embed_vectors_test=torch.tensor([]).cuda(), torch.tensor([]).cuda()
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader_train):
            a=batch['graphs']
            a_out=model(a.to('cuda'))
            embed_vectors_train=torch.cat((embed_vectors_train,a_out))   
        for i_batch, batch in enumerate(dataloader_test):
            a=batch['graphs']
            a_out=model(a.to('cuda'))
            embed_vectors_test=torch.cat((embed_vectors_test,a_out))
    embed_vectors_train=embed_vectors_train.cpu().detach().numpy()
    embed_vectors_test=embed_vectors_test.cpu().detach().numpy() 
    embed_dim=embed_vectors_test.shape[1]

    pickle.dump((embed_vectors_train, embed_vectors_test), open(f'features_GE_e{embed_dim}.pkl','wb'))
    print('Saved GE-vectors')

def gather_VGE_UE_vectors(dataloader_train, dataloader_test, model):
    #Gather all features
    print('Building GE vectors')
    embed_vectors_visual_train, embed_vectors_graph_train=torch.tensor([]).cuda(), torch.tensor([]).cuda()
    embed_vectors_visual_test, embed_vectors_graph_test=torch.tensor([]).cuda(), torch.tensor([]).cuda()
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader_train):
            out_visual, out_graph=model(batch['images'].to('cuda'), batch['graphs'].to('cuda'))
            embed_vectors_visual_train=torch.cat((embed_vectors_visual_train, out_visual))
            embed_vectors_graph_train=torch.cat((embed_vectors_graph_train, out_graph))  
        for i_batch, batch in enumerate(dataloader_test):
            out_visual, out_graph=model(batch['images'].to('cuda'), batch['graphs'].to('cuda'))
            embed_vectors_visual_test=torch.cat((embed_vectors_visual_test, out_visual))
            embed_vectors_graph_test=torch.cat((embed_vectors_graph_test, out_graph))  
    embed_vectors_visual_train=embed_vectors_visual_train.cpu().detach().numpy()
    embed_vectors_graph_train =embed_vectors_graph_train.cpu().detach().numpy()
    embed_vectors_visual_test=embed_vectors_visual_test.cpu().detach().numpy()
    embed_vectors_graph_test =embed_vectors_graph_test.cpu().detach().numpy()    
    embed_dim=embed_vectors_graph_train.shape[1]

    assert len(embed_vectors_visual_train)==len(embed_vectors_graph_train)==len(dataloader_train.dataset)
    assert len(embed_vectors_visual_test)==len(embed_vectors_graph_test)==len(dataloader_test.dataset)

    pickle.dump((embed_vectors_visual_train, embed_vectors_graph_train, embed_vectors_visual_test, embed_vectors_graph_test), open(f'features_VGE-UE_e{embed_dim}.pkl','wb'))
    print('Saved VGE-UE_vectors')    
  
'''
Goes from query-side embed-vectors to db-side embed vectors
Used for vectors from GE, VGE-UE and VGE
'''
def eval_GE_scoring(dataset_train, dataset_test, embedding_train, embedding_test ,top_k=(1,3,5,10), reduce_indices=None):
    assert len(embedding_train)==len(dataset_train) and len(embedding_test)==len(dataset_test)
    print(f'eval_GE_scoring(): # training: {len(dataset_train)}, # test: {len(dataset_test)}')
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
    pickle.dump(retrieval_dict, open(f'retrievals_GE.pkl','wb'))

    return evaluate_topK(pos_results, ori_results, scene_results)  

#TODO/CARE: norm before sum?
'''
Different ways of combining the the NetVLAD retrievals and GE retrievals
-Summing up the NV- and GE-distances (care: weighting cos-similarity vs. L2-distance)
-Combining both retrievals -> scene voting -> NetVLAD
'''
def eval_netvlad_embeddingVectors(dataset_train, dataset_test, netvlad_train, netvlad_test, embedding_train, embedding_test ,top_k=(1,3,5,10), combine='distance-sum'):
    assert combine in ('distance-sum','scene-voting->netvlad')
    print('\n eval_netvlad_embeddingVectors():', combine)

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

        netvlad_diffs  = np.linalg.norm( netvlad_train[:]  - netvlad_test[test_index] , axis=1 )
        scores=embedding_train@embedding_test[test_index]

        #Norm both for comparability
        netvlad_diffs = netvlad_diffs/np.max(np.abs(netvlad_diffs))
        scores= scores/np.max(np.abs(scores))
        assert len(netvlad_diffs)==len(scores)==len(dataset_train)

        pos_dists=np.linalg.norm(image_positions_train[:]-image_positions_test[test_index], axis=1) #CARE: also adds z-distance
        ori_dists=np.abs(image_orientations_train[:]-image_orientations_test[test_index])
        ori_dists=np.minimum(ori_dists, 2*np.pi-ori_dists)

        for k in top_k:
            if combine=='distance-sum': #TODO/CARE!
                combined_scores= scores + -1.0*netvlad_diffs 
                sorted_indices=np.argsort( -1.0*combined_scores)[0:k] #High->Low
            if combine=='scene-voting->netvlad':
                indices_netvlad=np.argsort(netvlad_diffs)[0:k] #Low->High
                indices_scenegraph=np.argsort(-1.0*scores)[0:k] #High->Low
                sorted_indices_netvlad,sorted_indices_ge=evaluation.utils.reduceIndices_sceneVoting(scene_names_train, indices_netvlad, indices_scenegraph)
                sorted_indices = sorted_indices_netvlad if len(sorted_indices_netvlad)>0 else sorted_indices_ge # Trust GE-indices if they are united enough to overrule NetVLAD, proved as best approach!

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
    pickle.dump(retrieval_dict, open(f'retrievals_netvlad_graph-embed_{combine}.pkl','wb'))

    return evaluate_topK(pos_results, ori_results, scene_results)          

def eval_VGE_UE_scoring():
    pass


'''
Evaluation between a NetVLAD-model and a model that predicts the NetVLAD vector cross-modal
distance-sum: predicts feature vectors separately, sums up the distances (both sides multi-modal)
mean-vector-test: uses the mean vector on the test-side, regular NetVLAD vector on the train side (query multi-modal)
mean-vector: uses the mean vectors on both sides (both sides multi-modal)
cross-modal: uses the predicted vectors on the test side, regular NetVLAD vectors on the train side (one modality per side)
embed-only: uses only the graph embeddings (both sides semantic-only, but "anchored" on NetVLAD)

-> Distance-Sum & Mean-Vector work the same and perform equal to pure-NV, mean-vector-test slightly worse, cross-modal bad, embed-only bad
-> VGE-NV does not precisely predict the NetVLAD vectors...
TODO: redo with smaller NetVLAD, higher Embed-Dim, combine by linear
'''
def eval_netvlad__VGE_NV(dataset_train, dataset_test, netvlad_train, netvlad_test, embedding_train, embedding_test ,top_k=(1,3,5,10), combine='distance-sum'):
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

#TODO: gather VGE-UE vectors (4 outputs)
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


    if 'gather' in sys.argv:
        #Gather Ge
        EMBED_DIM_GEOMETRIC=100
        geometric_embedding=GraphEmbedding(EMBED_DIM_GEOMETRIC)
        geometric_embedding_model_name='model_GraphEmbed_l3000_b12_g0.75_e100_sTrue_m1.0.pth'
        print('Model:',geometric_embedding_model_name)
        geometric_embedding.load_state_dict(torch.load('models/'+geometric_embedding_model_name))
        geometric_embedding.eval()
        geometric_embedding.cuda()         
        gather_GE_vectors(dataloader_train, dataloader_test, geometric_embedding)

        EMBED_DIM_GEOMETRIC=1024
        geometric_embedding=GraphEmbedding(EMBED_DIM_GEOMETRIC)
        geometric_embedding_model_name='model_GraphEmbed_l3000_b12_g0.75_e1024_sTrue_m1.0_lr0.005.pth'
        print('Model:',geometric_embedding_model_name)
        geometric_embedding.load_state_dict(torch.load('models/'+geometric_embedding_model_name))
        geometric_embedding.eval()
        geometric_embedding.cuda()         
        gather_GE_vectors(dataloader_train, dataloader_test, geometric_embedding)        

        #Gather VGE-UE
        EMBED_DIM_GEOMETRIC=1024
        vgg=create_image_model_vgg11()
        vge_ue_model=VisualGraphEmbedding(vgg, EMBED_DIM_GEOMETRIC).cuda()
        vge_ue_model_name='model_VGE-UE_l3000_b8_g0.75_e1024_sTrue_m0.5_lr0.0001.pth'
        vge_ue_model.load_state_dict(torch.load('models/'+vge_ue_model_name)); print('Model:',vge_ue_model_name)
        vge_ue_model.eval()
        vge_ue_model.cuda()
        gather_VGE_UE_vectors(dataloader_train, dataloader_test, vge_ue_model)

    if 'GE-match' in sys.argv:
        ge_vectors_filename='features_GE_e100.pkl'
        ge_vectors_train, ge_vectors_test=pickle.load(open('evaluation_res/'+ge_vectors_filename,'rb')); print('Using vectors',ge_vectors_filename)
        pos_results, ori_results, scene_results=eval_GE_scoring(dataset_train, dataset_test, ge_vectors_train, ge_vectors_test ,top_k=(1,3,5,10), reduce_indices=None)
        print(pos_results, ori_results, scene_results,'\n') 
        pos_results, ori_results, scene_results=eval_GE_scoring(dataset_train, dataset_test, ge_vectors_train, ge_vectors_test ,top_k=(1,3,5,10), reduce_indices='scene-voting')
        print(pos_results, ori_results, scene_results,'\n')         

        ge_vectors_filename='features_GE_e1024.pkl'
        ge_vectors_train, ge_vectors_test=pickle.load(open('evaluation_res/'+ge_vectors_filename,'rb')); print('Using vectors',ge_vectors_filename)
        pos_results, ori_results, scene_results=eval_GE_scoring(dataset_train, dataset_test, ge_vectors_train, ge_vectors_test ,top_k=(1,3,5,10))
        print(pos_results, ori_results, scene_results,'\n')         

    if 'NetVLAD+GE-match' in sys.argv:
        netvlad_vectors_filename='features_netvlad-S3D.pkl'
        netvlad_vectors_train,netvlad_vectors_test=pickle.load(open('evaluation_res/'+netvlad_vectors_filename,'rb')); print('Using vectors:', netvlad_vectors_filename)

        ge_vectors_filename='features_GE_e100.pkl'
        ge_vectors_train, ge_vectors_test=pickle.load(open('evaluation_res/'+ge_vectors_filename,'rb')); print('Using vectors',ge_vectors_filename)        

        pos_results, ori_results, scene_results=eval_netvlad_embeddingVectors(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, ge_vectors_train, ge_vectors_test ,top_k=(1,3,5,10), combine='distance-sum')
        print(pos_results, ori_results, scene_results,'\n')
        pos_results, ori_results, scene_results=eval_netvlad_embeddingVectors(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, ge_vectors_train, ge_vectors_test ,top_k=(1,3,5,10), combine='scene-voting->netvlad')
        print(pos_results, ori_results, scene_results,'\n')

    if 'VGE-UE-match' in sys.argv:
        vge_vectors_filename='features_VGE-UE_e1024.pkl'
        vge_vectors_visual_train, vge_vectors_graph_train, vge_vectors_visual_test, vge_vectors_graph_test=pickle.load(open('evaluation_res/'+vge_vectors_filename,'rb')); print('Using vectors',vge_vectors_filename)        

        print('Eval VGE-UE image-image')
        pos_results, ori_results, scene_results=eval_GE_scoring(dataset_train, dataset_test, vge_vectors_visual_train, vge_vectors_visual_test ,top_k=(1,3,5,10))
        print(pos_results, ori_results, scene_results,'\n')

        print('Eval VGE-UE graph-graph')
        pos_results, ori_results, scene_results=eval_GE_scoring(dataset_train, dataset_test, vge_vectors_graph_train, vge_vectors_graph_test ,top_k=(1,3,5,10)) 
        print(pos_results, ori_results, scene_results,'\n')

        print('Eval VGE-UE graph-image')
        pos_results, ori_results, scene_results=eval_GE_scoring(dataset_train, dataset_test, vge_vectors_visual_train, vge_vectors_graph_test ,top_k=(1,3,5,10))                
        print(pos_results, ori_results, scene_results,'\n')

    if 'NetVLAD+VGE-UE-match' in sys.argv:
        netvlad_vectors_filename='features_netvlad-S3D.pkl'
        netvlad_vectors_train,netvlad_vectors_test=pickle.load(open('evaluation_res/'+netvlad_vectors_filename,'rb')); print('Using vectors:', netvlad_vectors_filename)

        vge_vectors_filename='features_VGE-UE_e1024.pkl'
        vge_vectors_visual_train, vge_vectors_graph_train, vge_vectors_visual_test, vge_vectors_graph_test=pickle.load(open('evaluation_res/'+vge_vectors_filename,'rb')); print('Using vectors',vge_vectors_filename)        

        print('Eval NetVLAD + VGE-UE (graph->image)')
        pos_results, ori_results, scene_results=eval_netvlad_embeddingVectors(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, vge_vectors_visual_train, vge_vectors_graph_test ,top_k=(1,3,5,10), combine='distance-sum')
        print(pos_results, ori_results, scene_results,'\n')
        pos_results, ori_results, scene_results=eval_netvlad_embeddingVectors(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, vge_vectors_visual_train, vge_vectors_graph_test ,top_k=(1,3,5,10), combine='scene-voting->netvlad')
        print(pos_results, ori_results, scene_results,'\n')        

