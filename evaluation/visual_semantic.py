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

from visual_semantic.visual_semantic_embedding import VisualSemanticEmbedding, VisualSemanticEmbeddingNetVLAD, VisualSemanticEmbeddingCombined
from visual_semantic.semantic_embedding import SemanticEmbedding
from geometric.visual_graph_embedding import create_image_model_vgg11

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

def gather_VSE_vectors(dataloader_train, dataloader_test, model, model_name):
    assert model_name in ('VSE-UE','VSE-NV')
    #Gather all features
    print(f'Building {model_name} vectors')
    embed_vectors_visual_train, embed_vectors_semantic_train=torch.tensor([]).cuda(), torch.tensor([]).cuda()
    embed_vectors_visual_test, embed_vectors_semantic_test=torch.tensor([]).cuda(), torch.tensor([]).cuda()
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader_train):
            out_visual, out_semantic=model(batch['images'].to('cuda'), batch['captions'])
            embed_vectors_visual_train=torch.cat((embed_vectors_visual_train, out_visual))
            embed_vectors_semantic_train=torch.cat((embed_vectors_semantic_train, out_semantic))  
        for i_batch, batch in enumerate(dataloader_test):
            out_visual, out_semantic=model(batch['images'].to('cuda'), batch['captions'])
            embed_vectors_visual_test=torch.cat((embed_vectors_visual_test, out_visual))
            embed_vectors_semantic_test=torch.cat((embed_vectors_semantic_test, out_semantic))  
    embed_vectors_visual_train=embed_vectors_visual_train.cpu().detach().numpy()
    embed_vectors_semantic_train =embed_vectors_semantic_train.cpu().detach().numpy()
    embed_vectors_visual_test=embed_vectors_visual_test.cpu().detach().numpy()
    embed_vectors_semantic_test =embed_vectors_semantic_test.cpu().detach().numpy()    
    embed_dim=embed_vectors_semantic_train.shape[1]

    assert len(embed_vectors_visual_train)==len(embed_vectors_semantic_train)==len(dataloader_train.dataset)
    assert len(embed_vectors_visual_test)==len(embed_vectors_semantic_test)==len(dataloader_test.dataset)

    pickle.dump((embed_vectors_visual_train, embed_vectors_semantic_train, embed_vectors_visual_test, embed_vectors_semantic_test), open(f'features_{model_name}_e{embed_dim}.pkl','wb'))
    print(f'Saved {model_name}_vectors')     

def gather_VSE_CO_vectors(dataloader_train, dataloader_test, model):
    #Gather all features
    print('Building VSE-CO vectors')
    embed_vectors_train, embed_vectors_test=torch.tensor([]).cuda(), torch.tensor([]).cuda()
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader_train):
            out=model(batch['images'].to('cuda'), batch['captions'])
            embed_vectors_train=torch.cat((embed_vectors_train, out))
        for i_batch, batch in enumerate(dataloader_test):
            out=model(batch['images'].to('cuda'), batch['captions'])
            embed_vectors_test=torch.cat((embed_vectors_test, out)) 

    embed_vectors_train=embed_vectors_train.cpu().detach().numpy()
    embed_vectors_test =embed_vectors_test.cpu().detach().numpy()
    embed_dim=embed_vectors_train.shape[1]

    assert len(embed_vectors_train)==len(dataloader_train.dataset)
    assert len(embed_vectors_test)==len(dataloader_test.dataset)

    pickle.dump((embed_vectors_train, embed_vectors_test), open(f'features_VSE-CO_e{embed_dim}.pkl','wb'))
    print('Saved VSE-CO_vectors')   

'''
Goes from query-side embed-vectors to db-side embed vectors
Used for vectors from SE, VSE-UE and VSE-NV
'''
def eval_SE_scoring(dataset_train, dataset_test, embedding_train, embedding_test, similarity_measure ,top_k=(1,3,5,10), reduce_indices=None):
    assert len(embedding_train)==len(dataset_train) and len(embedding_test)==len(dataset_test)
    assert similarity_measure in ('cosine','l2')
    assert reduce_indices in (None, 'scene-voting')
    print(f'eval_SE_scoring(): # training: {len(dataset_train)}, # test: {len(dataset_test)}')
    print('Similarity measure:',similarity_measure,'Reduce indices:',reduce_indices) 

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
        
        if similarity_measure=='cosine':
            scores=embedding_train@embedding_test[test_index]
        if similarity_measure=='l2':
            scores= -1.0*np.linalg.norm( embedding_train-embedding_test[test_index], axis=1 )
        
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
    pickle.dump(retrieval_dict, open(f'retrievals_netvlad_semantic-embed_{combine}.pkl','wb'))

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
        EMBED_DIM_SEMANTIC=512
        semantic_embedding=SemanticEmbedding(dataset_train.get_known_words(),EMBED_DIM_SEMANTIC)
        semantic_embedding_model_name='model_SemanticEmbed2_l3000_b12_g0.75_e512_sTrue_m0.5_lr0.0005.pth'
        semantic_embedding.load_state_dict(torch.load('models/'+semantic_embedding_model_name)); print('Model:',semantic_embedding_model_name)
        semantic_embedding.eval()
        semantic_embedding.cuda()         
        gather_SE_vectors(dataloader_train, dataloader_test, semantic_embedding)

        #Gather VSE-UE
        EMBED_DIM_SEMANTIC=1024
        vgg=create_image_model_vgg11()
        vse_ue_model=VisualSemanticEmbedding(vgg, dataset_train.get_known_words(), EMBED_DIM_SEMANTIC).cuda()
        vse_ue_model_name='model_VSE-UE_l3000_b8_g0.75_e1024_sTrue_m0.5_lr0.05.pth'
        vse_ue_model.load_state_dict(torch.load('models/'+vse_ue_model_name)); print('Model:',vse_ue_model_name)
        vse_ue_model.eval()
        vse_ue_model.cuda()
        gather_VSE_vectors(dataloader_train, dataloader_test, vse_ue_model, 'VSE-UE')        

        #Gather VSE-NV
        EMBED_DIM_SEMANTIC=1024
        netvlad_model_name='model_netvlad_l3000_b6_g0.75_c8_a10.0.mdl'
        print('NetVLAD Model:',netvlad_model_name)
        netvlad_model=torch.load('models/'+netvlad_model_name)

        vse_nv_model=VisualSemanticEmbeddingNetVLAD(netvlad_model, dataset_train.get_known_words(), EMBED_DIM_SEMANTIC).cuda()
        vse_nv_model_name='model_VSE-NV_l3000_b8_g0.75_e1024_sTrue_m0.5_lr0.075.pth'
        vse_nv_model.load_state_dict(torch.load('models/'+vse_nv_model_name)); print('Model:',vse_nv_model_name)
        vse_nv_model.eval()
        vse_nv_model.cuda()
        gather_VSE_vectors(dataloader_train, dataloader_test, vse_nv_model, 'VSE-NV') 

        #Gather VSE-CO
        EMBED_DIM_SEMANTIC=1024               
        vgg=create_image_model_vgg11()
        vse_co_model=VisualSemanticEmbeddingCombined(vgg, dataset_train.get_known_words(), EMBED_DIM_SEMANTIC).cuda()
        vse_co_model_name='model_VSE-CO_l3000_b12_g0.75_e1024_sTrue_m0.5_lr0.00075.pth'
        vse_co_model.load_state_dict(torch.load('models/'+vse_co_model_name)); print('Model:',vse_co_model_name)
        vse_co_model.eval()
        vse_co_model.cuda()
        gather_VSE_CO_vectors(dataloader_train, dataloader_test, vse_co_model)                

    if 'SE-match' in sys.argv:
        se_vectors_filename='features_SE_e512.pkl'
        se_vectors_train, se_vectors_test=pickle.load(open('evaluation_res/'+se_vectors_filename,'rb')); print('Using vectors',se_vectors_filename)
        pos_results, ori_results, scene_results=eval_SE_scoring(dataset_train, dataset_test, se_vectors_train, se_vectors_test,'l2'    ,top_k=(1,3,5,10), reduce_indices=None)
        print(pos_results, ori_results, scene_results,'\n')    
        pos_results, ori_results, scene_results=eval_SE_scoring(dataset_train, dataset_test, se_vectors_train, se_vectors_test,'l2'    ,top_k=(1,3,5,10), reduce_indices='scene-voting')
        print(pos_results, ori_results, scene_results,'\n')          

    if 'NetVLAD+SE-match' in sys.argv:
        netvlad_vectors_filename='features_netvlad-S3D.pkl'
        netvlad_vectors_train,netvlad_vectors_test=pickle.load(open('evaluation_res/'+netvlad_vectors_filename,'rb')); print('Using vectors:', netvlad_vectors_filename)

        se_vectors_filename='features_SE_e512.pkl'
        se_vectors_train, se_vectors_test=pickle.load(open('evaluation_res/'+se_vectors_filename,'rb')); print('Using vectors',se_vectors_filename)        

        pos_results, ori_results, scene_results=eval_netvlad_embeddingVectors(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, se_vectors_train, se_vectors_test ,top_k=(1,3,5,10), combine='distance-sum')
        print(pos_results, ori_results, scene_results,'\n')
        pos_results, ori_results, scene_results=eval_netvlad_embeddingVectors(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, se_vectors_train, se_vectors_test ,top_k=(1,3,5,10), combine='scene-voting->netvlad')
        print(pos_results, ori_results, scene_results,'\n')     

    if 'VSE-UE-match' in sys.argv:
        vse_vectors_filename='features_VSE-UE_e1024.pkl'
        vse_vectors_visual_train, vse_vectors_caption_train, vse_vectors_visual_test, vse_vectors_caption_test=pickle.load(open('evaluation_res/'+vse_vectors_filename,'rb')); print('Using vectors',vse_vectors_filename)        

        print('Eval VSE-UE image-image')
        pos_results, ori_results, scene_results=eval_SE_scoring(dataset_train, dataset_test, vse_vectors_visual_train, vse_vectors_visual_test, 'l2', top_k=(1,3,5,10))
        print(pos_results, ori_results, scene_results,'\n')

        print('Eval VSE-UE caption-caption')
        pos_results, ori_results, scene_results=eval_SE_scoring(dataset_train, dataset_test, vse_vectors_caption_train, vse_vectors_caption_test, 'l2', top_k=(1,3,5,10)) 
        print(pos_results, ori_results, scene_results,'\n')

        print('Eval VSE-UE caption-image')
        pos_results, ori_results, scene_results=eval_SE_scoring(dataset_train, dataset_test, vse_vectors_visual_train, vse_vectors_caption_test, 'l2', top_k=(1,3,5,10))                
        print(pos_results, ori_results, scene_results,'\n')  

    if 'NetVLAD+VSE-UE-match' in sys.argv:
        netvlad_vectors_filename='features_netvlad-S3D.pkl'
        netvlad_vectors_train,netvlad_vectors_test=pickle.load(open('evaluation_res/'+netvlad_vectors_filename,'rb')); print('Using vectors:', netvlad_vectors_filename)

        vse_vectors_filename='features_VSE-UE_e1024.pkl'
        vse_vectors_visual_train, vse_vectors_caption_train, vse_vectors_visual_test, vse_vectors_caption_test=pickle.load(open('evaluation_res/'+vse_vectors_filename,'rb')); print('Using vectors',vse_vectors_filename)        

        print('Eval NetVLAD + VSE-UE (captions->image)')
        pos_results, ori_results, scene_results=eval_netvlad_embeddingVectors(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, vse_vectors_visual_train, vse_vectors_caption_test ,top_k=(1,3,5,10), combine='distance-sum')
        print(pos_results, ori_results, scene_results,'\n')
        pos_results, ori_results, scene_results=eval_netvlad_embeddingVectors(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, vse_vectors_visual_train, vse_vectors_caption_test ,top_k=(1,3,5,10), combine='scene-voting->netvlad')
        print(pos_results, ori_results, scene_results,'\n')                
              
    if 'VSE-NV-match' in sys.argv:
        vse_vectors_filename='features_VSE-NV_e1024.pkl'
        vse_vectors_visual_train, vse_vectors_caption_train, vse_vectors_visual_test, vse_vectors_caption_test=pickle.load(open('evaluation_res/'+vse_vectors_filename,'rb')); print('Using vectors',vse_vectors_filename)        

        print('Eval VSE-NV image-image')
        pos_results, ori_results, scene_results=eval_SE_scoring(dataset_train, dataset_test, vse_vectors_visual_train, vse_vectors_visual_test, 'l2', top_k=(1,3,5,10))
        print(pos_results, ori_results, scene_results,'\n')

        print('Eval VSE-NV caption-caption')
        pos_results, ori_results, scene_results=eval_SE_scoring(dataset_train, dataset_test, vse_vectors_caption_train, vse_vectors_caption_test, 'l2', top_k=(1,3,5,10)) 
        print(pos_results, ori_results, scene_results,'\n')

        print('Eval VSE-NV caption-image')
        pos_results, ori_results, scene_results=eval_SE_scoring(dataset_train, dataset_test, vse_vectors_visual_train, vse_vectors_caption_test, 'l2', top_k=(1,3,5,10))                
        print(pos_results, ori_results, scene_results,'\n')  

    if 'NetVLAD+VSE-NV-match' in sys.argv:
        netvlad_vectors_filename='features_netvlad-S3D.pkl'
        netvlad_vectors_train,netvlad_vectors_test=pickle.load(open('evaluation_res/'+netvlad_vectors_filename,'rb')); print('Using vectors:', netvlad_vectors_filename)

        vse_vectors_filename='features_VSE-NV_e1024.pkl'
        vse_vectors_visual_train, vse_vectors_caption_train, vse_vectors_visual_test, vse_vectors_caption_test=pickle.load(open('evaluation_res/'+vse_vectors_filename,'rb')); print('Using vectors',vse_vectors_filename)        

        print('Eval NetVLAD + VSE-NV (captions->image)')
        pos_results, ori_results, scene_results=eval_netvlad_embeddingVectors(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, vse_vectors_visual_train, vse_vectors_caption_test ,top_k=(1,3,5,10), combine='distance-sum')
        print(pos_results, ori_results, scene_results,'\n')
        pos_results, ori_results, scene_results=eval_netvlad_embeddingVectors(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, vse_vectors_visual_train, vse_vectors_caption_test ,top_k=(1,3,5,10), combine='scene-voting->netvlad')
        print(pos_results, ori_results, scene_results,'\n')                
       

    if 'VSE-CO-match' in sys.argv:
        vse_vectors_filename='features_VSE-CO_e1024.pkl'
        vse_vectors_train, vse_vectors_test=pickle.load(open('evaluation_res/'+vse_vectors_filename,'rb')); print('Using vectors',vse_vectors_filename)
        pos_results, ori_results, scene_results=eval_SE_scoring(dataset_train, dataset_test, vse_vectors_train, vse_vectors_test,'l2',top_k=(1,3,5,10), reduce_indices=None)
        print(pos_results, ori_results, scene_results,'\n')  
        pos_results, ori_results, scene_results=eval_SE_scoring(dataset_train, dataset_test, vse_vectors_train, vse_vectors_test,'l2',top_k=(1,3,5,10), reduce_indices='scene-voting')
        print(pos_results, ori_results, scene_results,'\n')        