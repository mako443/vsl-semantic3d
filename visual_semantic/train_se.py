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

from dataloading.data_loading import Semantic3dDatasetTriplet

from .semantic_embedding import SemanticEmbedding
from .visual_semantic_embedding import PairwiseRankingLoss

'''
Module to train a simple Semantic-Embedding model to score the similarity of captions (using no visual information)
'''

IMAGE_LIMIT=3000
BATCH_SIZE=6
LR_GAMMA=0.75
EMBED_DIM=100
SHUFFLE=True
DECAY=None #Tested, no decay here
MARGIN=0.5 #0.2: works, 0.4: increases loss, 1.0: TODO: acc, 2.0: loss unstable

print(f'Semantic Embedding training: image limit: {IMAGE_LIMIT} bs: {BATCH_SIZE} lr gamma: {LR_GAMMA} embed-dim: {EMBED_DIM} shuffle: {SHUFFLE} margin: {MARGIN}')

transform=transforms.Compose([
    #transforms.Resize((950,1000)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_set=Semantic3dDatasetTriplet('data/pointcloud_images_o3d_merged','train', transform=transform, image_limit=IMAGE_LIMIT, load_viewObjects=True, load_sceneGraphs=True, return_captions=True)
#Option: shuffle, pin_memory crashes on my system, 
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
        


