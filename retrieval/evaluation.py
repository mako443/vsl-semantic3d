import torch
import numpy as np
import cv2

from torch.utils.data import DataLoader
from torchvision import transforms

from retrieval import data_loading, networks
from retrieval.netvlad import NetVLAD, EmbedNet
from retrieval.utils import get_split_indices

#TODO: possibly make method accept w/ and w/o split (easy to just pass 2x same data loader?)
def evaluate_topK_dists(data_loader_train, data_loader_test, model, top_k=(1,5,10)):
    check_count=50
    print(f'# training: {len(data_loader_train.dataset)}, # test: {len(data_loader_test.dataset)}')

    netvlad_vectors_train, netvlad_vectors_test=torch.tensor([]).cuda(), torch.tensor([]).cuda()
    print('Building NetVLAD vectors...')
    # with torch.no_grad():
    #     for i_batch, batch in enumerate(data_loader_test): #loading p,n somewhat redundant
    #         a,p,n=batch
    #         a_out=model(a.cuda())
    #         netvlad_vectors_test=torch.cat((netvlad_vectors_test,a_out))
    #     for i_batch, batch in enumerate(data_loader_train):
    #         a,p,n=batch
    #         a_out=model(a.cuda())
    #         netvlad_vectors_train=torch.cat((netvlad_vectors_train,a_out))        

    # netvlad_vectors_train=netvlad_vectors_train.cpu().detach().numpy()
    # netvlad_vectors_test=netvlad_vectors_test.cpu().detach().numpy()

    #To sanity-check random vectors
    netvlad_vectors_train=np.random.rand(len(data_loader_train.dataset),2)
    netvlad_vectors_test=np.random.rand(len(data_loader_test.dataset),2)

    image_poses_train=data_loader_train.dataset.image_poses
    image_poses_test=data_loader_test.dataset.image_poses

    distance_sum={ k:0 for k in top_k }
    orientation_sum={ k:0 for k in top_k }
    scene_sum={ k:0 for k in top_k }

    print('evaluating random indices...')
    for i in range(check_count):
        #Pick a random test-image -> retrieve top-k train-images
        idx=np.random.choice(range(len(netvlad_vectors_test)))
        scene_name_gt=data_loader_test.dataset.get_scene_name(idx)

        netvlad_diffs=netvlad_vectors_train-netvlad_vectors_test[idx]
        netvlad_diffs=np.linalg.norm(netvlad_diffs,axis=1)
        #netvlad_diffs[idx]=np.inf

        sorted_indices=np.argsort(netvlad_diffs)

        location_dists=image_poses_train[:,0:3]-image_poses_test[idx,0:3]
        location_dists=np.linalg.norm(location_dists,axis=1)

        orientation_dists=np.abs(image_poses_train[:,3]-image_poses_test[idx,3]) 
        orientation_dists=np.minimum(orientation_dists,2*np.pi-orientation_dists)      

        #Calculating mean loc./ori. dists only for correct scenes
        for k in top_k:
            scene_correct=np.array([scene_name_gt == data_loader_train.dataset.get_scene_name(retrieved_index) for retrieved_index in sorted_indices[0:k]])
            topk_loc_dists=location_dists[sorted_indices[0:k]]
            topk_ori_dists=orientation_dists[sorted_indices[0:k]]
            
            if np.sum(scene_correct)>0:
                distance_sum[k]   +=np.mean( topk_loc_dists[scene_correct==True] )
                orientation_sum[k]+=np.mean( topk_ori_dists[scene_correct==True] )
                scene_sum[k]      +=np.mean(scene_correct)

            #distance_sum[k]+=np.mean( location_dists[sorted_indices[0:k]] )   
            #orientation_sum[k]+=np.mean( orientation_dists[sorted_indices[0:k]] )
            #scene_sum[k]+= np.mean(np.array( [scene_name == data_loader.dataset.get_scene_name(retrieved_index) for retrieved_index in sorted_indices[0:k]] )) #Correctness sanity-checked with random vecs


    for k in top_k:
        distance_sum[k]/=check_count
        orientation_sum[k]/=check_count
        scene_sum[k]/=check_count
        distance_sum[k],orientation_sum[k], scene_sum[k]=np.float16(distance_sum[k]),np.float16(orientation_sum[k]),np.float16(scene_sum[k]) #Make numbers more readable

    return distance_sum, orientation_sum, scene_sum

if __name__ == "__main__":
    #CARE: make sure model and data_set are the same as in training!
    IMAGE_LIMIT=None
    BATCH_SIZE=6
    LR_GAMMA=0.75
    NUM_CLUSTERS=8
    TEST_SPLIT=4

    transform=transforms.Compose([
        #transforms.Resize((950,1000)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_indices, test_indices=get_split_indices(TEST_SPLIT, 4*240)

    data_set_train=data_loading.Semantic3dData('data/pointcloud_images', transform=transform, image_limit=IMAGE_LIMIT, split_indices=train_indices) #CARE: correct split
    data_loader_train=DataLoader(data_set_train, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False)

    data_set_test=data_loading.Semantic3dData('data/pointcloud_images', transform=transform, image_limit=IMAGE_LIMIT, split_indices=test_indices) #CARE: correct split
    data_loader_test=DataLoader(data_set_test, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False)    

    encoder=networks.get_encoder_resnet18()
    encoder.requires_grad_(False) #Don't train encoder
    netvlad_layer=NetVLAD(num_clusters=NUM_CLUSTERS, dim=512, alpha=10.0)

    model=EmbedNet(encoder, netvlad_layer)
    model.load_state_dict(torch.load('model_4scenes_simpleSplit4.pth'))
    model=model.cuda()
    model.eval()

    res=evaluate_topK_dists(data_loader_train, data_loader_test, model)
    print(res)
        
