import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torchvision.models
import string
import random
import os
import numpy as np
import matplotlib.pyplot as plt

from retrieval.utils import get_split_indices
from dataloading.data_loading import Semantic3dDataset
from visual_semantic.visual_semantic_embedding import PairwiseRankingLoss
from .visual_graph_embedding import VisualGraphEmbedding, create_image_model_vgg11

'''
Visual Graph Embedding training

TODO:
-Very small
-Train big, evaluate
-NetVLAD backbone
-train w/ TripletMarginLoss?

-NetVLAD backbone, train all together w/ TripletMarginLoss
'''

IMAGE_LIMIT=60
BATCH_SIZE=2 #12 gives memory error, 8 had more loss than 6?
LR_GAMMA=0.75
TEST_SPLIT=4
EMBED_DIM=300
SHUFFLE=True
MARGIN=1.0 #0.2: works, 0.4: increases loss, 1.0: TODO: acc, 2.0: loss unstable

print(f'image limit: {IMAGE_LIMIT} bs: {BATCH_SIZE} lr gamma: {LR_GAMMA} test-split: {TEST_SPLIT} embed-dim: {EMBED_DIM} shuffle: {SHUFFLE} margin: {MARGIN}')

transform=transforms.Compose([
    #transforms.Resize((950,1000)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_indices, test_indices=get_split_indices(TEST_SPLIT, 3000)

data_set=Semantic3dDataset('data/pointcloud_images_o3d_merged', transform=transform, image_limit=IMAGE_LIMIT, split_indices=None, load_viewObjects=True, load_sceneGraphs=True, return_graph_data=True)
#Option: shuffle, pin_memory crashes on my system, CARE: shuffle for PairWiseRankingLoss(!)
data_loader=DataLoader(data_set, batch_size=BATCH_SIZE, num_workers=2, pin_memory=False, shuffle=SHUFFLE) 

loss_dict={}
best_loss=np.inf
best_model=None

for lr in (2e-2,1e-2,5e-3):
    print('\n\nlr: ',lr)

    vgg=create_image_model_vgg11()
    model=VisualGraphEmbedding(vgg, EMBED_DIM).cuda()

    criterion=PairwiseRankingLoss(margin=MARGIN)
    optimizer=optim.Adam(model.parameters(), lr=lr) #Adam is ok for PyG
    scheduler=optim.lr_scheduler.ExponentialLR(optimizer,LR_GAMMA)   

    if type(criterion)==PairwiseRankingLoss: assert SHUFFLE==True 

    loss_dict[lr]=[]
    for epoch in range(3):
        epoch_loss_sum=0.0
        for i_batch, batch in enumerate(data_loader):
            
            optimizer.zero_grad()
            print(batch)
            
            out_visual, out_graph=model(batch['images'].cuda(), batch['graphs'].to('cuda'))

            loss=criterion(out_visual, out_graph)
            loss.backward()
            optimizer.step()

            l=loss.cpu().detach().numpy()
            epoch_loss_sum+=l
            #print(f'\r epoch {epoch} loss {l}',end='')
        
        scheduler.step()

        epoch_avg_loss = epoch_loss_sum/(i_batch+1)
        print(f'epoch {epoch} final avg-loss {epoch_avg_loss}')
        loss_dict[lr].append(epoch_avg_loss)

    #Now using loss-avg of last epoch!
    if epoch_avg_loss<best_loss:
        best_loss=epoch_avg_loss
        best_model=model

print('\n----')           
model_name=f'model_vge_l{IMAGE_LIMIT}_b{BATCH_SIZE}_g{LR_GAMMA:0.2f}_e{EMBED_DIM}_s{SHUFFLE}_m{MARGIN}_split{TEST_SPLIT}.pth'
print('Saving best model',model_name)
torch.save(best_model.state_dict(),model_name)

for k in loss_dict.keys():
    l=loss_dict[k]
    line, = plt.plot(l)
    line.set_label(k)
plt.legend()
#plt.show()
plt.savefig(f'loss_vge_l{IMAGE_LIMIT}_b{BATCH_SIZE}_g{LR_GAMMA:0.2f}_e{EMBED_DIM}_s{SHUFFLE}_m{MARGIN}_split{TEST_SPLIT}.png')    
