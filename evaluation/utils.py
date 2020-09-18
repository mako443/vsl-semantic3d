import pickle
import numpy as np
import cv2
import os


#TODO
#Sanity check by always selecting up to top-3 retrievals
#Evaluate for simple NetVLAD
def reduceIndices_sceneVoting(scene_names_train, indices0, indices1=None):
    if indices1 is None:
        indices0=np.array(indices0)
        indices_scene_names=scene_names_train[indices0]
        most_frequent_scene_name=max(set(indices_scene_names), key=list(indices_scene_names).count)
        return indices0[indices_scene_names==most_frequent_scene_name]
    else:
        indices0,indices1=np.array(indices0),np.array(indices1)
        indices0_scene_names,indices1_scene_names=scene_names_train[indices0],scene_names_train[indices1]
        indices_scene_names=np.hstack((indices0_scene_names,indices1_scene_names))
        most_frequent_scene_name=max(set(indices_scene_names), key=list(indices_scene_names).count)
        return indices0[indices0_scene_names==most_frequent_scene_name], indices1[indices1_scene_names==most_frequent_scene_name]



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

if __name__=='__main__':
    names=['s0','s1','s2']
    indices0=[0,1,2]
    indices1=[0,1,2]