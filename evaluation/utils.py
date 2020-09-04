import pickle
import numpy as np
import cv2
import os

def evaluate_topK(pos_results, ori_results, scene_results):
    top_k=list(pos_results.keys())

    pos,ori,scene={},{},{}
    for k in top_k:
        assert len(pos_results[k])==len(ori_results[k])==len(scene_results[k])
        pos_results_valid=[result for result in pos_results[k] if result is not None]
        ori_results_valid=[result for result in ori_results[k] if result is not None]
        pos[k]=np.float16(np.mean( pos_results_valid )) if len(pos_results_valid)>0 else np.nan
        ori[k]=np.float16(np.mean( ori_results_valid )) if len(ori_results_valid)>0 else np.nan
        scene[k]=np.float16(np.mean(scene_results[k]))

    return pos, ori, scene      

#One index in test that has 3 exact matches in train and 7 non-matches in train
def generate_sanity_check_dataset():
    netvlad_test=np.array([0.0,0.0,0.0]).reshape((1,3))
    pos_test=np.array([0.0,0.0,0.0]).reshape((1,3))
    ori_test=np.array([0.0])
    scene_names_test=['test-scene',]

    netvlad_train=10.0*np.ones((10,3))
    netvlad_train[0:3,:]=[0.0,0.0,0.0] #3 exact matches

    pos_train=10.0*np.ones((10,3))
    pos_train[0:3,:]=[0.0,0.0,0.0] #3 exact matches

    ori_train=np.pi*np.ones(10)
    ori_train[0:3]=0.0

    scene_names_train=np.array(['train-scene' for i in range(10)])
    scene_names_train[:]='test-scene' #3 exact matches

    return netvlad_train, netvlad_test, pos_train, pos_test, ori_train, ori_test, scene_names_train, scene_names_test