import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
import cv2
import psutil
import matplotlib.pyplot as plt

from retrieval import data_loading, networks
from retrieval.netvlad import NetVLAD, EmbedNet

'''
TODO:
-basic Resnet18+NetVLAD -> loss doesn't drop much
-check feature-alikeness vs. Aachen/DeepLoc (norm w/ norms) -> ✖ can't confirm - possibly distorted by a/p alikeness
-enforce closest anchor -> ✓ Yes, seems to help. Before the was sometimes barely overlap.
-check for reasons the loss is not dropping -> Positive pairs to close

-overfit 200 images, evaluate top-k dists, train on more scenes
-go to 12-16 angles or 1:1 images, take near-enough positives again

-bigger encoder (FCN-Resnet101), too big for my GPU
-ggf. train segmentation model (check avg p/n feature alikeness before/after)


MODELS:
resnet18, 200 images: ({1: 1.5981454108206739, 5: 3.4949463674080286, 10: 5.17567078938991}, {1: 0.0, 5: 0.3612831551628262, 10: 0.4162610266006476})
'''

IMAGE_LIMIT=None
BATCH_SIZE=8
LR_GAMMA=0.75
NUM_CLUSTERS=8

print(f'image limit: {IMAGE_LIMIT} bs: {BATCH_SIZE} lr gamma: {LR_GAMMA} clusters: {NUM_CLUSTERS}')

transform=transforms.Compose([
    transforms.Resize((950,1000)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_set=data_loading.Semantic3dData('data/pointcloud_images', transform=transform, image_limit=IMAGE_LIMIT)
data_loader=DataLoader(data_set, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False)

loss_dict={}
best_loss=np.inf
best_model=None

for lr in (5e-1,2.5e-1,1e-1,):
    print('\n\nlr: ',lr)
    encoder=networks.get_encoder_resnet18()
    encoder.requires_grad_(False) #Don't train encoder
    netvlad_layer=NetVLAD(num_clusters=NUM_CLUSTERS, dim=512, alpha=10.0)

    model=EmbedNet(encoder, netvlad_layer).cuda()

    # encoder=encoder.cuda()
    # a,p,n=next(iter(data_loader))
    # a_out=encoder(a.cuda())
    # p_out=encoder(p.cuda())
    # n_out=encoder(n.cuda())
    # norm=torch.norm(a_out)
    # p_diff=torch.norm(a_out-p_out)
    # n_diff=torch.norm(a_out-n_out)
    # print(p_diff)
    # print(n_diff)
    # print(p_diff/n_diff)

    # quit()

    criterion=nn.TripletMarginLoss(margin=1.0)
    optimizer=optim.Adam(model.parameters(), lr=lr)    
    scheduler=optim.lr_scheduler.ExponentialLR(optimizer,LR_GAMMA)    

    loss_dict[lr]=[]
    for epoch in range(6):
        epoch_loss_sum=0.0
        for i_batch, batch in enumerate(data_loader):
            a,p,n=batch        
            
            optimizer.zero_grad()
            a_out=model(a.cuda())
            p_out=model(p.cuda())
            n_out=model(n.cuda())

            loss=criterion(a_out,p_out,n_out)
            loss.backward()
            optimizer.step()

            if (i_batch-1)%1==0:
                l=loss.cpu().detach().numpy()
                epoch_loss_sum+=l
                #print(f'\r epoch {epoch} loss {l}',end='')
        
        scheduler.step()

        epoch_loss_sum/= i_batch+1
        print(f'\n epoch {epoch} avg-loss {epoch_loss_sum}',end='')
        loss_dict[lr].append(epoch_loss_sum)

    if loss_dict[lr][-1]<best_loss:
        best_loss=loss_dict[lr][-1]
        best_model=model

print('\n----')           
print('Saving best model')
torch.save(best_model.state_dict(),'last_best_model.pth')


for k in loss_dict.keys():
    l=loss_dict[k]
    line, = plt.plot(l)
    line.set_label(k)
plt.legend()
#plt.show()
plt.savefig(f'loss_l{IMAGE_LIMIT}_b{BATCH_SIZE}_g{LR_GAMMA:0.2f}.png')    