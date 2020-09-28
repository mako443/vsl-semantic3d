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
<<<<<<< HEAD
=======
import sys
>>>>>>> 704fca48fe7fad1927ded32f47ca5b94ae9a21d1
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.data import DataLoader #Use the PyG DataLoader

from dataloading.data_loading import Semantic3dDatasetTriplet

from .semantic_embedding import SemanticEmbedding
from .visual_semantic_embedding import PairwiseRankingLoss

'''
Module to train a simple Semantic-Embedding model to score the similarity of captions (using no visual information)
'''

IMAGE_LIMIT=3000
<<<<<<< HEAD
BATCH_SIZE=6
LR_GAMMA=0.75
EMBED_DIM=100
SHUFFLE=True
DECAY=None #Tested, no decay here
MARGIN=0.5 #0.2: works, 0.4: increases loss, 1.0: TODO: acc, 2.0: loss unstable

print(f'Semantic Embedding training: image limit: {IMAGE_LIMIT} bs: {BATCH_SIZE} lr gamma: {LR_GAMMA} embed-dim: {EMBED_DIM} shuffle: {SHUFFLE} margin: {MARGIN}')
=======
BATCH_SIZE=12
LR_GAMMA=0.75
EMBED_DIM=512 # 100 performed worse than GE (which had )
SHUFFLE=True
#DECAY=None #Tested, no decay here
MARGIN=0.5 #0.2: works, 0.4: increases loss, 1.0: TODO: acc, 2.0: loss unstable

#CAPTURE arg values
LR=float(sys.argv[-1])

print(f'Semantic Embedding training: image limit: {IMAGE_LIMIT} bs: {BATCH_SIZE} lr gamma: {LR_GAMMA} embed-dim: {EMBED_DIM} shuffle: {SHUFFLE} margin: {MARGIN} LR: {LR}')
>>>>>>> 704fca48fe7fad1927ded32f47ca5b94ae9a21d1

transform=transforms.Compose([
    #transforms.Resize((950,1000)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_set=Semantic3dDatasetTriplet('data/pointcloud_images_o3d_merged','train', transform=transform, image_limit=IMAGE_LIMIT, load_viewObjects=True, load_sceneGraphs=True, return_captions=True)
#Option: shuffle, pin_memory crashes on my system, 
<<<<<<< HEAD
#data_loader=DataLoader(data_set, batch_size=BATCH_SIZE, num_workers=2, pin_memory=False, shuffle=SHUFFLE) 
known_words=data_set.get_known_words()

# model=SemanticEmbedding(np.hstack((known_words,'CLASS0','CLASS1')),EMBED_DIM)
# model.cuda()
# out=model(('In buildings','In buildings'))
# quit()

sanity_captions=[]
sanity_classes=[]
for i in range(100):
    cap=list(np.random.choice(known_words,size=np.random.randint(5,10)))
    cap_class=np.random.choice([0,1])
    if cap_class==0:
        cap.insert(np.random.randint(0,len(cap)+1), 'CLASS0')
    else:
        cap.insert(np.random.randint(0,len(cap)+1), 'CLASS1')
    cap=' '.join(cap)
    sanity_captions.append(cap)
    sanity_classes.append(cap_class)

# sanity_captions=np.array(sanity_captions)
# sanity_classes=np.array(sanity_classes)


for lr in (1e-2,1e-3,1e-4):
    model=SemanticEmbedding(np.hstack((known_words,'CLASS0','CLASS1')),EMBED_DIM)
    #model.train()
    model.cuda()    

    criterion = torch.nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(), lr=lr) #Using SGD for packed Embedding

    for epoch in range(100):
        epoch_losses=[]
        for i in range(10):
            caps=sanity_captions[i*2:(i+1)*2]
            classes=sanity_classes[i*2:(i+1)*2]
            optimizer.zero_grad()
            
            out=model(caps)
            loss=criterion(out, torch.from_numpy(classes).cuda())

            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            
        if epoch%20==0: 
            print(np.mean(epoch_losses))
            #_,pred=model(graph).max(dim=1)
            #acc= torch.sum(pred==graph.y).item()/len(out)
            #print(loss, acc)
        


=======
data_loader=DataLoader(data_set, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=SHUFFLE) 

loss_dict={}
best_loss=np.inf
best_model=None

for lr in (5e-4*8, 5e-4*4, 5e-4, 5e-4/4, 5e-4/8):
#for lr in (LR,):
    print('\n\nlr: ',lr)

    model=SemanticEmbedding(data_set.get_known_words(),EMBED_DIM)
    model.cuda()

    criterion=nn.TripletMarginLoss(margin=MARGIN)
    optimizer=optim.Adam(model.parameters(), lr=lr) #Adam is ok for PyG | Apparently also for packed_sequence!
    scheduler=optim.lr_scheduler.ExponentialLR(optimizer,LR_GAMMA)   

    loss_dict[lr]=[]
    for epoch in range(6):
        epoch_loss_sum=0.0
        for i_batch, batch in enumerate(data_loader):
            
            optimizer.zero_grad()
            #print(batch)
            
            a_out=model(batch['captions_anchor'])
            p_out=model(batch['captions_positive'])
            n_out=model(batch['captions_negative'])

            loss=criterion(a_out,p_out,n_out)
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
model_name=f'model_SemanticEmbed2_l{IMAGE_LIMIT}_b{BATCH_SIZE}_g{LR_GAMMA:0.2f}_e{EMBED_DIM}_s{SHUFFLE}_m{MARGIN}_lr{LR}.pth'
print('Saving best model',model_name)
torch.save(best_model.state_dict(),model_name)

for k in loss_dict.keys():
    l=loss_dict[k]
    line, = plt.plot(l)
    line.set_label(k)
plt.gca().set_ylim(bottom=0.0) #Set the bottom to 0.0
plt.legend()
#plt.show()
plt.savefig(f'loss_SemanticEmbed2_l{IMAGE_LIMIT}_b{BATCH_SIZE}_g{LR_GAMMA:0.2f}_e{EMBED_DIM}_s{SHUFFLE}_m{MARGIN}_lr{LR}.png')    
>>>>>>> 704fca48fe7fad1927ded32f47ca5b94ae9a21d1
