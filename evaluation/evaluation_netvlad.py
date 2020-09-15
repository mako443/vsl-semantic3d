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

from retrieval import networks
from retrieval.netvlad import NetVLAD, EmbedNet

from evaluation.utils import evaluate_topK, generate_sanity_check_dataset

def netvlad_retrieval(data_loader_train, data_loader_test, model, top_k=(1,3,5,10), check_count='all'):
    print(f'#Check: {check_count}, # training: {len(data_loader_train.dataset)}, # test: {len(data_loader_test.dataset)}')

    retrieval_dict={}

    print('Building NetVLAD vectors...')
    netvlad_vectors_train, netvlad_vectors_test=torch.tensor([]).cuda(), torch.tensor([]).cuda()

    with torch.no_grad():
        for i_batch, batch in enumerate(data_loader_test):
            a=batch
            a_out=model(a.cuda())
            netvlad_vectors_test=torch.cat((netvlad_vectors_test,a_out))
        for i_batch, batch in enumerate(data_loader_train):
            a=batch
            a_out=model(a.cuda())
            netvlad_vectors_train=torch.cat((netvlad_vectors_train,a_out))        
    netvlad_vectors_train=netvlad_vectors_train.cpu().detach().numpy()
    netvlad_vectors_test=netvlad_vectors_test.cpu().detach().numpy()

    image_positions_train, image_orientations_train = data_loader_train.dataset.image_positions, data_loader_train.dataset.image_orientations
    image_positions_test, image_orientations_test = data_loader_test.dataset.image_positions, data_loader_test.dataset.image_orientations
    scene_names_train = data_loader_train.dataset.image_scene_names
    scene_names_test  = data_loader_test.dataset.image_scene_names

    #Sanity check
    #netvlad_vectors_train, netvlad_vectors_test, image_positions_train, image_positions_test, image_orientations_train, image_orientations_test, scene_names_train, scene_names_test=generate_sanity_check_dataset()

    pos_results  ={k:[] for k in top_k}
    ori_results  ={k:[] for k in top_k}
    scene_results={k:[] for k in top_k}

    if check_count=='all':
        print('evaluating all indices...')
        check_indices=np.arange(len(data_loader_test.dataset))
    else:
        print('evaluating random indices...')
        check_indices=np.random.randint(len(data_loader_test.dataset), size=check_count)
        
    for idx in check_indices:
        scene_name_gt=scene_names_test[idx]

        netvlad_diffs=netvlad_vectors_train-netvlad_vectors_test[idx]
        netvlad_diffs=np.linalg.norm(netvlad_diffs,axis=1)   

        sorted_indices=np.argsort(netvlad_diffs) 
        pos_dists=np.linalg.norm(image_positions_train[:]-image_positions_test[idx], axis=1) #CARE: also adds z-distance
        ori_dists=np.abs(image_orientations_train[:]-image_orientations_test[idx])
        ori_dists=np.minimum(ori_dists, 2*np.pi-ori_dists)

        retrieval_dict[idx]=sorted_indices[0:np.max(top_k)]

        for k in top_k:
            #scene_correct=np.array([scene_name_gt == data_loader_train.dataset.get_scene_name(retrieved_index) for retrieved_index in sorted_indices[0:k]])
            scene_correct=np.array([scene_name_gt == scene_names_train[retrieved_index] for retrieved_index in sorted_indices[0:k]])
            topk_pos_dists=pos_dists[sorted_indices[0:k]]
            topk_ori_dists=ori_dists[sorted_indices[0:k]]    

            #Append the average pos&ori. errors *for the cases that the scene was hit*
            pos_results[k].append( np.mean( topk_pos_dists[scene_correct==True]) if np.sum(scene_correct)>0 else None )
            ori_results[k].append( np.mean( topk_ori_dists[scene_correct==True]) if np.sum(scene_correct)>0 else None )
            scene_results[k].append( np.mean(scene_correct) ) #Always append the scene-scores
    
    assert len(pos_results[k])==len(ori_results[k])==len(scene_results[k])==len(check_indices)

    print('Saving retrieval results...')
    pickle.dump(retrieval_dict, open('retrievals_cambridge_NetVLAD.pkl','wb'))

    return evaluate_topK(pos_results, ori_results, scene_results)

if __name__ == "__main__":
    if 'pretrained-netvlad' in sys.argv:
        import pytorch_NetVlad.main
        print(model)
        # assert os.path.isfile('../pytorch-NetVlad/netvlad_pittsburgh.pth')
        # model=torch.load('../pytorch-NetVlad/netvlad_pittsburgh.pth')
        # model.eval()
        # input=torch.rand(10,3,1000,1000).cuda()
        # image_encoding = model.encoder(input)
        # vlad_encoding = model.pool(image_encoding) 
        # print('shape',vlad_encoding.shape)

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
