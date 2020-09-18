import pickle
import numpy as np
import cv2
import os

from dataloading.data_loading import Semantic3dDataset, Semantic3dDatasetTriplet
from retrieval import networks
from retrieval.netvlad import NetVLAD, EmbedNet
import torch
from torchvision import transforms

def get_top1_scene_misses(retrieval_dict, dataset_train, dataset_test):
    results={} # {query_idx: db_idx}
    for query_idx in retrieval_dict.keys():
        db_idx=retrieval_dict[query_idx][0]
        query_scene=dataset_test.image_scene_names[query_idx]
        db_scene=dataset_train.image_scene_names[db_idx]

        if query_scene!=db_scene:
            results[query_idx]=db_idx

    return results

def get_top5_all_scene_misses(retrieval_dict, dataset_train, dataset_test):
    results={} # {query_idx: [db_idx0, .., db_idx4] }
    for query_idx in retrieval_dict.keys():
        query_scene=dataset_test.image_scene_names[query_idx]
        db_indices=retrieval_dict[query_idx][0:5]
        db_scenes=dataset_train.image_scene_names[db_indices]  
        if not np.any( db_scenes==query_scene ):
            results[query_idx]=db_indices
    return results

#Scene correct, then biggest position errors
def get_biggest_top1_position_erros(retrieval_dict, dataset_train, dataset_test, num_results=10):
    query_indices=list(retrieval_dict.keys())
    db_indices   =[retrieval_dict[query_idx][0] for query_idx in query_indices]
    assert len(query_indices)==len(db_indices)

    position_erros=[ dataset_test.image_positions[ query_indices[i] ] - dataset_train.image_positions[ db_indices[i] ] for i in range(len(query_indices)) ]
    position_erros=np.linalg.norm(position_erros,axis=1)
    assert len(position_erros)==len(query_indices)

    sorted_indices=np.argsort(-1*position_erros) #Sort descending

    results= {query_indices[i]: db_indices[i] for i in sorted_indices[0:num_results] }
    print(f'Top {num_results} position errors:', position_erros[sorted_indices[0:num_results]])

    return results

def view_cases(cases, dataset_train, dataset_test, num_cases=10):
    if len(cases)<=num_cases: query_indices=list(cases.keys())
    else: query_indices= np.random.choice(list(cases.keys()), size=num_cases)
    
    for i,query_idx in enumerate(query_indices):
        db_indices=cases[query_idx]
        print('Showing',query_idx, db_indices)
        img_query=  cv2.cvtColor(np.asarray(dataset_test[query_idx]), cv2.COLOR_RGB2BGR)
        if type(db_indices) in (list, np.ndarray):
            img_db   =  cv2.cvtColor( np.hstack(( [np.asarray(dataset_train[db_idx]) for db_idx in db_indices] )) , cv2.COLOR_RGB2BGR)
            img_query, img_db= cv2.resize(img_query, None, fx=0.5, fy=0.5), cv2.resize(img_db, None, fx=0.5, fy=0.5)
        else:
            img_db   =  cv2.cvtColor(np.asarray(dataset_train[db_indices]), cv2.COLOR_RGB2BGR)
        cv2.imshow("query", img_query)
        cv2.imshow("db",    img_db)
        cv2.imwrite(f'cases_{i}.png', np.hstack(( img_query, img_db )) )
        cv2.waitKey()

#Results sanity-checked (NetVLAD) ✓
if __name__ == "__main__":
    IMAGE_LIMIT=3000
    BATCH_SIZE=6
    NUM_CLUSTERS=8
    TEST_SPLIT=4
    ALPHA=10.0   

    dataset_train=Semantic3dDataset('data/pointcloud_images_o3d_merged','train',transform=None, image_limit=IMAGE_LIMIT)
    dataset_test =Semantic3dDataset('data/pointcloud_images_o3d_merged','test', transform=None, image_limit=IMAGE_LIMIT)

retrieval_dict=pickle.load(open('retrievals/'+'retrievals_NV_SG-Match.pkl','rb'))
assert len(dataset_test)==len(retrieval_dict)

cases=get_biggest_top1_position_erros(retrieval_dict, dataset_train, dataset_test)
print(f'#Cases: {len(cases)}')

view_cases(cases, dataset_train, dataset_test)