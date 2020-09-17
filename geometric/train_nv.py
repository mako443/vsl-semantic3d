import torch
import torch.nn as nn
import torch.optim as optim
#from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torchvision.models
import string
import random
import os
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.data import DataLoader #Use the PyG DataLoader

from retrieval.utils import get_split_indices
from dataloading.data_loading import Semantic3dDataset
from visual_semantic.visual_semantic_embedding import PairwiseRankingLoss
from .visual_graph_embedding import VisualGraphEmbedding,VisualGraphEmbeddingNetVLAD, create_image_model_vgg11, create_image_model_netvlad

'''
Visual Graph Embedding training (NetVLAD backbone)

TODO:
-Weight decay ✓
-Sanity: NV-vectors from encode_image() and pure NetVLAD are same ✓

-Sanity: Can the model predict NV-outputs?!
'''

IMAGE_LIMIT=120
BATCH_SIZE=6 #12 gives memory error, 8 had more loss than 6?
LR_GAMMA=0.75
EMBED_DIM=300
SHUFFLE=True
DECAY=0.001 #This decay proved best
MARGIN=1.0 #0.2: works, 0.4: increases loss, 1.0: TODO: acc, 2.0: loss unstable

print(f'VGE-NV (naive) training: image limit: {IMAGE_LIMIT} bs: {BATCH_SIZE} lr gamma: {LR_GAMMA} embed-dim: {EMBED_DIM} shuffle: {SHUFFLE} margin: {MARGIN} decay: {DECAY}')

transform=transforms.Compose([
    #transforms.Resize((950,1000)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_set=Semantic3dDataset('data/pointcloud_images_o3d_merged','train', transform=transform, image_limit=IMAGE_LIMIT, load_viewObjects=True, load_sceneGraphs=True, return_graph_data=True)
#Option: shuffle, pin_memory crashes on my system, CARE: shuffle for PairWiseRankingLoss(!)
data_loader=DataLoader(data_set, batch_size=BATCH_SIZE, num_workers=2, pin_memory=False, shuffle=SHUFFLE) 

loss_dict={}
best_loss=np.inf
best_model=None

for lr in (1e-2,5e-3,1e-3,5e-4,1e-4):
    print('\n\nlr: ',lr)

    netvlad=create_image_model_netvlad()
    model=VisualGraphEmbeddingNetVLAD(netvlad, EMBED_DIM)
    model.cuda()

    #Sanity check: can it predict NV vectors?!
    model.load_state_dict(torch.load('model_vgeNV_l120_b6_g0.75_e300_sTrue_m1.0_d0.001.pth'))
    model.cuda()
    netvlad.cuda()

    for batch in data_loader:
        out_visual, out_graph=model(batch['images'].cuda(), batch['graphs'].to('cuda'))
        out_visual, out_graph=out_visual.cpu().detach().numpy(), out_graph.cpu().detach().numpy()
        print('g',np.linalg.norm( out_visual - out_graph ) / np.linalg.norm( out_visual) )
        print('min',np.min(out_visual),np.min(out_graph))
        print('max',np.max(out_visual),np.max(out_graph))
    quit()

    #criterion=nn.MSELoss()
    criterion=nn.L1Loss() #CARE: using L1 loss for testing
    optimizer=optim.Adam(model.parameters(), lr=lr, weight_decay=DECAY) #Adam is ok for PyG
    scheduler=optim.lr_scheduler.ExponentialLR(optimizer,LR_GAMMA)   

    loss_dict[lr]=[]
    for epoch in range(8):
        epoch_loss_sum=0.0
        for i_batch, batch in enumerate(data_loader):
            
            optimizer.zero_grad()
            #print(batch)
            
            out_visual, out_graph=model(batch['images'].cuda(), batch['graphs'].to('cuda'))

            loss=criterion(out_visual, out_graph)
            loss.backward()
            optimizer.step()

            l=loss.cpu().detach().numpy()
            epoch_loss_sum+=l
            #print(f'\r epoch {epoch} loss {l}',end='')
        
        scheduler.step()

        epoch_avg_loss = epoch_loss_sum/(i_batch+1)
        print(f'epoch {epoch} final avg-loss {epoch_avg_loss} vec-sum {np.sum( np.abs(out_graph.cpu().detach().numpy()),axis=1)}')
        loss_dict[lr].append(epoch_avg_loss)

    #Now using loss-avg of last epoch!
    if epoch_avg_loss<best_loss:
        best_loss=epoch_avg_loss
        best_model=model

print('\n----')           
model_name=f'model_vgeNV_l{IMAGE_LIMIT}_b{BATCH_SIZE}_g{LR_GAMMA:0.2f}_e{EMBED_DIM}_s{SHUFFLE}_m{MARGIN}_d{DECAY}.pth'
print('Saving best model',model_name)
torch.save(best_model.state_dict(),model_name)

for k in loss_dict.keys():
    l=loss_dict[k]
    line, = plt.plot(l)
    line.set_label(k)
plt.gca().set_ylim(bottom=0.0) #Set the bottom to 0.0
plt.legend()
#plt.show()
plt.savefig(f'loss_vgeNV_l{IMAGE_LIMIT}_b{BATCH_SIZE}_g{LR_GAMMA:0.2f}_e{EMBED_DIM}_s{SHUFFLE}_m{MARGIN}_d{DECAY}.png')    
