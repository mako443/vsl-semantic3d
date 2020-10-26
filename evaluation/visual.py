import pickle
import numpy as np
import cv2
import os
import torch
import sys
#from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import DataLoader #Use the PyG DataLoader

from dataloading.data_cambridge import CambridgeDataset
from dataloading.data_loading import Semantic3dDataset

from retrieval import networks
from retrieval.netvlad import NetVLAD, EmbedNet

from evaluation.utils import evaluate_topK, generate_sanity_check_dataset
import evaluation.utils

def gather_netvlad_vectors(dataloader_train, dataloader_test, model, features_name):
    #Gather all features
    print('Building NetVLAD vectors,',features_name)
    netvlad_vectors_train, netvlad_vectors_test=torch.tensor([]).cuda(), torch.tensor([]).cuda()
    with torch.no_grad():
        for i_batch, batch in enumerate(dataloader_test):
            a=batch
            a_out=netvlad_model(a.cuda())
            netvlad_vectors_test=torch.cat((netvlad_vectors_test,a_out))
        for i_batch, batch in enumerate(dataloader_train):
            a=batch
            a_out=netvlad_model(a.cuda())
            netvlad_vectors_train=torch.cat((netvlad_vectors_train,a_out))   
    netvlad_vectors_train=netvlad_vectors_train.cpu().detach().numpy()
    netvlad_vectors_test=netvlad_vectors_test.cpu().detach().numpy() 

    pickle.dump((netvlad_vectors_train, netvlad_vectors_test), open(f'features_netvlad_{features_name}.pkl','wb'))
    print('Saved NetVLAD-vectors,',features_name)

#TODO
#Sanity-check same performance ✓
#Sanity-check top-3 combine ✓
#Evaluate scene-voting ✓ works!
def eval_netvlad_retrieval(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, top_k=(1,3,5,10), reduce_indices=None):
    assert reduce_indices in (None,'scene-voting','scene-voting-double-k')
    print(f'eval_netvlad_retrieval(): # training: {len(dataset_train)}, # test: {len(dataset_test)}')
    print('Reduce indices:',reduce_indices)

    retrieval_dict={}

    image_positions_train, image_orientations_train = dataset_train.image_positions, dataset_train.image_orientations
    image_positions_test, image_orientations_test = dataset_test.image_positions, dataset_test.image_orientations
    scene_names_train = dataset_train.image_scene_names
    scene_names_test  = dataset_test.image_scene_names

    #Sanity check
    #netvlad_vectors_train, netvlad_vectors_test, image_positions_train, image_positions_test, image_orientations_train, image_orientations_test, scene_names_train, scene_names_test=generate_sanity_check_dataset()

    pos_results  ={k:[] for k in top_k}
    ori_results  ={k:[] for k in top_k}
    scene_results={k:[] for k in top_k}
        
    test_indices=np.arange(len(dataset_test))    
    for test_index in test_indices:
        scene_name_gt=scene_names_test[test_index]

        netvlad_diffs=netvlad_vectors_train-netvlad_vectors_test[test_index]
        netvlad_diffs=np.linalg.norm(netvlad_diffs,axis=1)   

        pos_dists=np.linalg.norm(image_positions_train[:]-image_positions_test[test_index], axis=1) #CARE: also adds z-distance
        ori_dists=np.abs(image_orientations_train[:]-image_orientations_test[test_index])
        ori_dists=np.minimum(ori_dists, 2*np.pi-ori_dists)

        for k in top_k:
            if reduce_indices is None:
                sorted_indices=np.argsort(netvlad_diffs)[0:k] #Sanity still same result ✓
            if reduce_indices =='scene-voting':
                sorted_indices=np.argsort(netvlad_diffs)[0:k]
                sorted_indices=evaluation.utils.reduceIndices_sceneVoting(scene_names_train, sorted_indices)
            if reduce_indices =='scene-voting-double-k':
                sorted_indices=np.argsort(netvlad_diffs)[0:k]
                sorted_indices_voting=np.argsort(netvlad_diffs)[ k : 2*k ] # Take another k top retrievals just for scene-voting to compare to combined models
                sorted_indices_topK,sorted_indices_doubleK=evaluation.utils.reduceIndices_sceneVoting(scene_names_train, sorted_indices, sorted_indices_voting)
                sorted_indices = sorted_indices_topK if len(sorted_indices_topK)>0 else sorted_indices_doubleK #Same logic as in visual_geometric, trust next indices if they are "united" enough to over-rule top-k

            if k==np.max(top_k): retrieval_dict[test_index]=sorted_indices                

            scene_correct=np.array([scene_name_gt == scene_names_train[retrieved_index] for retrieved_index in sorted_indices])
            topk_pos_dists=pos_dists[sorted_indices]
            topk_ori_dists=ori_dists[sorted_indices]    

            #Append the average pos&ori. errors *for the cases that the scene was hit*
            pos_results[k].append( np.mean( topk_pos_dists[scene_correct==True]) if np.sum(scene_correct)>0 else None )
            ori_results[k].append( np.mean( topk_ori_dists[scene_correct==True]) if np.sum(scene_correct)>0 else None )
            scene_results[k].append( np.mean(scene_correct) ) #Always append the scene-scores
    
    assert len(pos_results[k])==len(ori_results[k])==len(scene_results[k])==len(test_indices)

    print('Saving retrieval results...')
    pickle.dump(retrieval_dict, open('retrievals_NV-S3D.pkl','wb'))

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

    dataset_train=Semantic3dDataset('data/pointcloud_images_o3d_merged','train',transform=transform, image_limit=IMAGE_LIMIT, load_viewObjects=True, load_sceneGraphs=True)
    dataset_test =Semantic3dDataset('data/pointcloud_images_o3d_merged','test', transform=transform, image_limit=IMAGE_LIMIT, load_viewObjects=True, load_sceneGraphs=True)

    dataloader_train=DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False) #CARE: put shuffle off
    dataloader_test =DataLoader(dataset_test , batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False)        

    if 'gather' in sys.argv:
        image_encoder=networks.get_encoder_resnet18()
        image_encoder.requires_grad_(False) #Don't train encoder
        netvlad_layer=NetVLAD(num_clusters=NUM_CLUSTERS, dim=512, alpha=ALPHA)
        netvlad_model=EmbedNet(image_encoder, netvlad_layer)

        netvlad_model_name='model_netvlad_l3000_b6_g0.75_c8_a10.0.pth'
        print('Model:',netvlad_model_name)
        netvlad_model.load_state_dict(torch.load('models/'+netvlad_model_name))
        netvlad_model.eval()
        netvlad_model.cuda()
        gather_netvlad_vectors(dataloader_train, dataloader_test, netvlad_model)

    if 'gather-occ' in sys.argv:
        dataset_train=Semantic3dDataset('data/pointcloud_images_o3d_merged_occ','train',transform=transform, image_limit=IMAGE_LIMIT, load_viewObjects=True, load_sceneGraphs=True)
        dataset_test =Semantic3dDataset('data/pointcloud_images_o3d_merged_occ','test', transform=transform, image_limit=IMAGE_LIMIT, load_viewObjects=True, load_sceneGraphs=True)

        dataloader_train=DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False) #CARE: put shuffle off
        dataloader_test =DataLoader(dataset_test , batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False)     

        image_encoder=networks.get_encoder_resnet18()
        image_encoder.requires_grad_(False) #Don't train encoder
        netvlad_layer=NetVLAD(num_clusters=NUM_CLUSTERS, dim=512, alpha=ALPHA)
        netvlad_model=EmbedNet(image_encoder, netvlad_layer)

        netvlad_model_name='model_netvlad_l3000_b8_g0.75_c8_a10.0_m0.5_o0.3_lr0.02.pth'
        print('Model:',netvlad_model_name)
        netvlad_model.load_state_dict(torch.load('models/'+netvlad_model_name))
        netvlad_model.eval()
        netvlad_model.cuda()
        gather_netvlad_vectors(dataloader_train, dataloader_test, netvlad_model,'Occ-Occ_m0.5_o0.3')   

        netvlad_model_name='model_netvlad_l3000_b8_g0.75_c8_a10.0_m0.5_o0.2_lr-1.0.pth'
        print('Model:',netvlad_model_name)
        netvlad_model.load_state_dict(torch.load('models/'+netvlad_model_name))
        netvlad_model.eval()
        netvlad_model.cuda()
        gather_netvlad_vectors(dataloader_train, dataloader_test, netvlad_model,'Occ-Occ_m0.5_o0.2')   

        netvlad_model_name='model_netvlad_l3000_b8_g0.75_c8_a10.0_m0.5_o0.4_lr-1.0.pth'
        print('Model:',netvlad_model_name)
        netvlad_model.load_state_dict(torch.load('models/'+netvlad_model_name))
        netvlad_model.eval()
        netvlad_model.cuda()
        gather_netvlad_vectors(dataloader_train, dataloader_test, netvlad_model,'Occ-Occ_m0.5_o0.4')                        

    if 'netvlad' in sys.argv:
        features_name='features_netvlad-S3D.pkl'
        netvlad_vectors_train,netvlad_vectors_test=pickle.load(open('evaluation_res/'+features_name,'rb')); print(f'Using features {features_name}')
        pos_results, ori_results, scene_results=eval_netvlad_retrieval(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, top_k=(1,3,5,10), reduce_indices=None)
        print(pos_results, ori_results, scene_results)
        pos_results, ori_results, scene_results=eval_netvlad_retrieval(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, top_k=(1,3,5,10), reduce_indices='scene-voting')
        print(pos_results, ori_results, scene_results)
        pos_results, ori_results, scene_results=eval_netvlad_retrieval(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, top_k=(1,3,5,10), reduce_indices='scene-voting-double-k')
        print(pos_results, ori_results, scene_results)    

        features_name='features_netvlad_Base-Occ.pkl'
        netvlad_vectors_train,netvlad_vectors_test=pickle.load(open('evaluation_res/'+features_name,'rb')); print(f'Using features {features_name}')        
        pos_results, ori_results, scene_results=eval_netvlad_retrieval(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, top_k=(1,3,5,10), reduce_indices=None)
        print(pos_results, ori_results, scene_results)
        pos_results, ori_results, scene_results=eval_netvlad_retrieval(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, top_k=(1,3,5,10), reduce_indices='scene-voting')
        print(pos_results, ori_results, scene_results)
        pos_results, ori_results, scene_results=eval_netvlad_retrieval(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, top_k=(1,3,5,10), reduce_indices='scene-voting-double-k')
        print(pos_results, ori_results, scene_results)      

        features_name='features_netvlad_Occ-Occ_m0.5_o0.2.pkl'
        netvlad_vectors_train,netvlad_vectors_test=pickle.load(open('evaluation_res/'+features_name,'rb')); print(f'Using features {features_name}')        
        pos_results, ori_results, scene_results=eval_netvlad_retrieval(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, top_k=(1,3,5,10), reduce_indices=None)
        print(pos_results, ori_results, scene_results)
        pos_results, ori_results, scene_results=eval_netvlad_retrieval(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, top_k=(1,3,5,10), reduce_indices='scene-voting')
        print(pos_results, ori_results, scene_results) 

        features_name='features_netvlad_Occ-Occ_m0.5_o0.3.pkl'
        netvlad_vectors_train,netvlad_vectors_test=pickle.load(open('evaluation_res/'+features_name,'rb')); print(f'Using features {features_name}')        
        pos_results, ori_results, scene_results=eval_netvlad_retrieval(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, top_k=(1,3,5,10), reduce_indices=None)
        print(pos_results, ori_results, scene_results)
        pos_results, ori_results, scene_results=eval_netvlad_retrieval(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, top_k=(1,3,5,10), reduce_indices='scene-voting')
        print(pos_results, ori_results, scene_results) 

        features_name='features_netvlad_Occ-Occ_m0.5_o0.4.pkl'
        netvlad_vectors_train,netvlad_vectors_test=pickle.load(open('evaluation_res/'+features_name,'rb')); print(f'Using features {features_name}')        
        pos_results, ori_results, scene_results=eval_netvlad_retrieval(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, top_k=(1,3,5,10), reduce_indices=None)
        print(pos_results, ori_results, scene_results)
        pos_results, ori_results, scene_results=eval_netvlad_retrieval(dataset_train, dataset_test, netvlad_vectors_train, netvlad_vectors_test, top_k=(1,3,5,10), reduce_indices='scene-voting')
        print(pos_results, ori_results, scene_results)                              

    if 'netvlad-cambridge' in sys.argv:
        IMAGE_LIMIT=3000
        BATCH_SIZE=6
        NUM_CLUSTERS=8
        ALPHA=10.0
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])  

        data_set_train=CambridgeDataset('data_cambridge','train',transform=transform)
        data_set_test =CambridgeDataset('data_cambridge','test', transform=transform)

        data_loader_train=DataLoader(data_set_train, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False) #CARE: put shuffle off
        data_loader_test =DataLoader(data_set_test , batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False)        

        print('## Evaluation: S3D-trained NetVLAD on Cambridge')
        encoder=networks.get_encoder_resnet18()
        encoder.requires_grad_(False) #Don't train encoder
        netvlad_layer=NetVLAD(num_clusters=NUM_CLUSTERS, dim=512, alpha=ALPHA)
        model=EmbedNet(encoder, netvlad_layer)

        model_name='model_netvlad_l3000_b6_g0.75_c8_a10.0.pth'
        model.load_state_dict(torch.load('models/'+model_name))
        model.eval()
        model.cuda()
        print('Model:',model_name)

        pos_results, ori_results, scene_results=netvlad_retrieval(data_loader_train, data_loader_test, model)
        print(pos_results, ori_results, scene_results)    
