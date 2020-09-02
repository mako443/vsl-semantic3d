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

from retrieval import networks
from retrieval.netvlad import NetVLAD, EmbedNet
from retrieval.utils import get_split_indices
from dataloading.data_loading import Semantic3dDatasetTriplet

'''
TODO:
-basic Resnet18+NetVLAD -> loss doesn't drop much
-check feature-alikeness vs. Aachen/DeepLoc (norm w/ norms) -> ✖ can't confirm - possibly distorted by a/p alikeness
-enforce closest anchor -> ✓ Yes, seems to help. Before the was sometimes barely overlap.
-check for reasons the loss is not dropping -> Positive pairs to close
-overfit 200 images, evaluate top-k dists, train on more scenes ✓
-Train&eval 4 scenes 3:2 aspect no split and 3-1 split: ✓ scene-retrieval good, high 
-nächstes Mal r=0? -> Yes, bzw eh via Open3D ✓

-Redo splitting w/ disjoint trajectories, less locations, random angles

-calc. average location dist, compare: 

-Splitting&near-enough positives: bigger fov, check point-size, enough overlap w/ 12 angles
-Compare training w/ nearest vs. random positive-anchor

-bigger encoder (FCN-Resnet101), too big for my GPU
-ggf. train segmentation model (check avg p/n feature alikeness before/after)

-Find pairs via Open3D visible points?

MODELS:
resnet18, 200 images            : ({1: 1.598, 5: 3.49, 10: 5.17}, {1: 0.0,   5: 0.361, 10: 0.416})
resnet18, 480i, 50-50 neg. idx  : ({1: 5.681, 5: 8.44, 10: 9.98}, {1: 0.701, 5: 0.955, 10: 1.087}, {1: 0.92, 5: 0.868, 10: 0.80})
resnet18, 480i, same scene neg. :

resnet18, 480i, 3:2 , no split  : ({1: 4.805, 5: 7.617, 10: 9.23}, {1: 0.2932, 5: 0.463, 10: 0.609}, {1: 1.0, 5: 0.964, 10: 0.95})
same, 3-1 split test->train ret.: ({1: 4.55, 5: 5.336, 10: 6.27}, {1: 1.184, 5: 1.171, 10: 1.283}, {1: 1.0, 5: 0.988, 10: 0.976}) -> CARE: Higher ori. error because the nearest ones are "taken"
same, random                    : ({1: 5.887, 5: 10.17, 10: 12.414}, {1: 0.4817, 5: 1.416, 10: 1.522}, {1: 0.34, 5: 0.288, 10: 0.276}) #CARE: random can be quite volatile

----New scenes----

resnet18, 1scene, 3:2, 3-1 split:

'''

IMAGE_LIMIT=50
BATCH_SIZE=6
LR_GAMMA=0.75
NUM_CLUSTERS=8
TEST_SPLIT=4

print(f'image limit: {IMAGE_LIMIT} bs: {BATCH_SIZE} lr gamma: {LR_GAMMA} clusters: {NUM_CLUSTERS} test-split: {TEST_SPLIT}')

transform=transforms.Compose([
    #transforms.Resize((950,1000)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_indices, test_indices=get_split_indices(TEST_SPLIT, 600+0)

data_set=Semantic3dDatasetTriplet('data/pointcloud_images_o3d_merged_1scene', transform=transform, image_limit=IMAGE_LIMIT, split_indices=train_indices, load_viewObjects=False, load_sceneGraphs=False)
data_loader=DataLoader(data_set, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=False)

loss_dict={}
best_loss=np.inf
best_model=None

#for lr in (5e-2,1e-2, 5e-3):
for lr in (1e-1,5e-2,1e-2):
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
    for epoch in range(5):
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

            l=loss.cpu().detach().numpy()
            epoch_loss_sum+=l
            print(f'\r epoch {epoch} loss {l}',end='')
        
        scheduler.step()

        epoch_avg_loss = epoch_loss_sum/(i_batch+1)
        print(f'\n epoch {epoch} avg-loss {epoch_avg_loss}')
        loss_dict[lr].append(epoch_avg_loss)

    #Now using loss-avg of last epoch!
    if epoch_avg_loss<best_loss:
        best_loss=epoch_avg_loss
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
plt.savefig(f'loss_l{IMAGE_LIMIT}_b{BATCH_SIZE}_g{LR_GAMMA:0.2f}_split{TEST_SPLIT}.png')    