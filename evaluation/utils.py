import pickle
import numpy as np
import cv2
import os

def evaluate_topK(pos_results, ori_results, scene_results):
    top_k=list(pos_results.keys())

    pos,ori,scene={},{},{}
    for k in top_k:
        assert len(pos_results[k])==len(ori_results[k])==len(scene_results[k])
        pos[k]=np.float16(np.mean( [result for result in pos_results[k] if result is not None] ))
        ori[k]=np.float16(np.mean( [result for result in ori_results[k] if result is not None] ))
        scene[k]=np.float16(np.mean(scene_results[k]))

    return pos, ori, scene      