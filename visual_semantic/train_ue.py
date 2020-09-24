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
import sys
import numpy as np
import cv2
import psutil
import matplotlib.pyplot as plt

from .visual_semantic_embedding import VisualSemanticEmbedding, PairwiseRankingLoss
from retrieval import networks
from retrieval.netvlad import NetVLAD, EmbedNet
from retrieval.utils import get_split_indices
from dataloading.data_loading import Semantic3dDataset

'''
Visual Semantic Embedding training

TODO:
-Train big, evaluate -> performance weak ✖
-Train with other contrastive and/or shuffle? -> shuffle helps! ✓

-Verify w/ image limit and no split
-Use other backbone?
-Use other captions?
-Train w/ TripletMarginLoss?
'''

'''
10 scenes, m0.2, e300, no shuffle: bad
10 scenes, m1.0, e300, w/ shuffle: {1: 48.44, 3: 41.7, 5: 39.94, 10: 45.8} {1: 1.824, 3: 1.677, 5: 1.593, 10: 1.58} {1: 0.26, 3: 0.2467, 5: 0.262, 10: 0.289}
10 scenes, m1.0, e100, w/ shuffle: {1: 44.84, 3: 38.2, 5: 36.53, 10: 36.53} {1: 1.775, 3: 1.652, 5: 1.616, 10: 1.605} {1: 0.23, 3: 0.3018, 5: 0.335, 10: 0.294}
'''

IMAGE_LIMIT=420
BATCH_SIZE=8 #12 gives memory error, 8 had more loss than 6?
LR_GAMMA=0.75
EMBED_DIM=1024
SHUFFLE=True
MARGIN=0.5 #0.2: works, 0.4: increases loss, 1.0: TODO: acc, 2.0: loss unstable

#Capture arguments
LR=float(sys.argv[-1])

print(f'VSE-UE training: image limit: {IMAGE_LIMIT} bs: {BATCH_SIZE} lr gamma: {LR_GAMMA} embed-dim: {EMBED_DIM} shuffle: {SHUFFLE} margin: {MARGIN} lr:{LR}')

transform=transforms.Compose([
    #transforms.Resize((950,1000)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_set=Semantic3dDataset('data/pointcloud_images_o3d_merged','train', transform=transform, image_limit=IMAGE_LIMIT, load_viewObjects=True, load_sceneGraphs=True, return_captions=True)
data_loader=DataLoader(data_set, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=SHUFFLE) #Option: shuffle, Care: pin_memory!

loss_dict={}
best_loss=np.inf
best_model=None

for lr in (5e-4,1e-4,5e-5):
    print('\n\nlr: ',lr)

    vgg=torchvision.models.vgg11(pretrained=True)
    for i in [4,5,6]: vgg.classifier[i]=nn.Identity()     #Remove layers after the 4096 features Linear layer

    model=VisualSemanticEmbedding(vgg, data_set.get_known_words(), EMBED_DIM).cuda()

    criterion=PairwiseRankingLoss(margin=MARGIN)
    optimizer=optim.SGD(model.parameters(), lr=lr) #Using SGD for packed Embedding
    scheduler=optim.lr_scheduler.ExponentialLR(optimizer,LR_GAMMA)    

    loss_dict[lr]=[]
    for epoch in range(5):
        epoch_loss_sum=0.0
        for i_batch, batch in enumerate(data_loader):
            optimizer.zero_grad()
            x,v=model(batch['images'].cuda(),batch['captions'])            

            loss=criterion(x, v)
            #TODO: clip grad norm?
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
model_name=f'model_VSE-UE_l{IMAGE_LIMIT}_b{BATCH_SIZE}_g{LR_GAMMA:0.2f}_e{EMBED_DIM}_s{SHUFFLE}_m{MARGIN}_l{LOSS}_lr{LR}.pth'
print('Saving best model',model_name)
torch.save(best_model.state_dict(),model_name)

for k in loss_dict.keys():
    l=loss_dict[k]
    line, = plt.plot(l)
    line.set_label(k)
plt.gca().set_ylim(bottom=0.0) #Set the bottom to 0.0
plt.legend()
#plt.show()
plt.savefig(f'loss_VSE-UE_l{IMAGE_LIMIT}_b{BATCH_SIZE}_g{LR_GAMMA:0.2f}_e{EMBED_DIM}_s{SHUFFLE}_m{MARGIN}_l{LOSS}_lr{LR}.png')    
