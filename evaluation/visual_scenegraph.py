import pickle
import numpy as np
import cv2
import os
import torch
import sys
#from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models
import torch.nn as nn

from torch_geometric.data import DataLoader #Use the PyG DataLoader

from dataloading.data_loading import Semantic3dDataset, Semantic3dDatasetTriplet
from retrieval import networks
from retrieval.netvlad import NetVLAD, EmbedNet

from semantic.imports import SceneGraph, SceneGraphObject, ViewObject
from semantic.scene_graph_cluster3d_scoring import score_sceneGraph_to_viewObjects_nnRels

def gather_sceneGraph_scores(dataset_train, dataset_test):
    print(f'gather_sceneGraph_scores(): # training: {len(dataset_train)}, # test: {len(dataset_test)}')

    score_dict={} # {test-idx: {train_idx: score} }

    for test_idx in range(len(dataset_test)):
        score_dict[test_idx]={}
        scene_graph=dataset_test.view_scenegraphs[test_idx]
        for train_idx in range(len(dataset_train)):
            score,_=score_sceneGraph_to_viewObjects_nnRels(scene_graph, dataset_train.view_objects[train_idx], unused_factor=0.5)
            score_dict[test_idx][train_idx]=score  

        assert len(score_dict[test_idx])==len(dataset_train)
    assert len(score_dict)==len(dataset_test)

    print('Saving SG-scores...')
    pickle.dump(score_dict, open('scenegraph_scores.pkl','wb'))

if __name__ == "__main__":
    IMAGE_LIMIT=3000
    dataset_train=Semantic3dDataset('data/pointcloud_images_o3d_merged','train',transform=None, image_limit=IMAGE_LIMIT, load_viewObjects=True, load_sceneGraphs=True)
    dataset_test =Semantic3dDataset('data/pointcloud_images_o3d_merged','test', transform=None, image_limit=IMAGE_LIMIT, load_viewObjects=True, load_sceneGraphs=True)    

    if 'gather' in sys.argv:
        gather_sceneGraph_scores(dataset_train, dataset_test)