import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import os

#TODO: how to handle "distance" between scenes?
class Semantic3dData(Dataset):
    def __init__(self, dirpath_main, transform=None, image_limit=None):
        assert os.path.isdir(dirpath_main)

        self.dirpath_main=dirpath_main
        self.transform=transform

        self.positive_thresh=np.array((7.5,2*np.pi/12*1.01))
        self.negative_thresh=np.array((10,np.pi / 2))

        self.scene_names=[folder_name for folder_name in os.listdir(dirpath_main) if os.path.isdir(os.path.join(dirpath_main,folder_name))]
        print('Semantic3dData scene names: ',self.scene_names)

        self.image_paths=[]
        self.image_poses=np.array([],np.float32).reshape(0,6)
        #Read all image names
        for scene_name in self.scene_names:
            scene_path=os.path.join(dirpath_main,scene_name,'rgb')
            self.image_paths.extend( [os.path.join(scene_path, image_name) for image_name in sorted(os.listdir(scene_path))] ) #Care: sort the image-names to agree with the poses

            poses_path=os.path.join(dirpath_main,scene_name,'poses.npy')
            assert os.path.isfile(poses_path)
            self.image_poses=np.vstack((self.image_poses,np.fromfile(poses_path).reshape((-1,6))))# [x,y,z,phi,theta,r]

        assert len(self.image_paths)==len(self.image_poses)

        if image_limit and image_limit>0:
            self.image_limit=image_limit
        else:
            self.image_limit=None

    #Retrieves a triplet with [idx] as anchor, currently all from same scene TODO: chance to take negative from other scene randomly
    def __getitem__(self,anchor_index):
        # count=0
        # for scene_name in self.scene_names:
        #     if count+len(self.images_dict[scene_name]) <= anchor_index:
        #         count+=len(self.images_dict[scene_name])
        #     else:
        #         anchor_index=anchor_index-count
        #         break

        scene_name=self.get_scene_name(anchor_index) #The scene-name for the image at anchor_index

        # scene_poses=self.poses_dict[scene_name]
        # location_dists=scene_poses[:,0:3]-scene_poses[anchor_index,0:3]
        # location_dists=np.linalg.norm(location_dists,axis=1)

        # orientation_dists=np.abs(scene_poses[:,3]-scene_poses[anchor_index,3]) #CARE: Currently just considering phi
        # orientation_dists=np.minimum(orientation_dists,2*np.pi-orientation_dists)

        #CARE: not considering different scenes yet
        location_dists=self.image_poses[:,0:3]-self.image_poses[anchor_index,0:3] 
        location_dists=np.linalg.norm(location_dists,axis=1)

        orientation_dists=np.abs(self.image_poses[:,3]-self.image_poses[anchor_index,3]) #CARE: Currently just considering phi
        orientation_dists=np.minimum(orientation_dists,2*np.pi-orientation_dists)


        #Find positive index (currently using minimum orientation_dist)
        location_dists[anchor_index]=np.inf
        orientation_dists[anchor_index]=np.inf
        indices= (location_dists<self.positive_thresh[0]) & (orientation_dists<self.positive_thresh[1]) & (np.core.defchararray.find(self.image_paths, scene_name)!=-1) #location&ori. dists small enough, same scene
        assert np.sum(indices)>0
        indices=np.argwhere(indices==True).flatten()
        if len(indices)<2: print('Warning: only 1 pos. index choice')
        positive_index=np.random.choice(indices)
        
        #min_index=np.argmin(orientation_dists[indices]) #the sub-index of indicies corresponding to the smalles orientation-dist
        #positive_index=indices[min_index]

        #Find negative index
        #if np.random.choice([True,False]): #Pick an index of the same scene
        if True:
            location_dists[anchor_index]=0
            orientation_dists[anchor_index]=0
            indices= (location_dists>=self.negative_thresh[0]) & (orientation_dists>=self.negative_thresh[1]) & (np.core.defchararray.find(self.image_paths, scene_name)!=-1)# loc.&ori. dists big enough, same scene
            assert np.sum(indices)>0
            indices=np.argwhere(indices==True).flatten()
            negative_index=np.random.choice(indices)
        # else: #Pick an index of another scene
        #     indices=np.flatnonzero(np.core.defchararray.find(self.image_paths, scene_name)==-1) #
        #     negative_index=np.random.choice(indices)

        #print(anchor_index, positive_index, negative_index)

        anchor  =  Image.open(self.image_paths[anchor_index]).convert('RGB')
        positive = Image.open(self.image_paths[positive_index]).convert('RGB')
        negative = Image.open(self.image_paths[negative_index]).convert('RGB')
        #pose=self.poses_dict[scene_name][anchor_index]

        #image_rgb=Image.open(path_rgb)

        if self.transform:
            anchor, positive, negative = self.transform(anchor),self.transform(positive),self.transform(negative)

        return anchor,positive, negative

    def __len__(self):
        if self.image_limit:
            return self.image_limit
        else:
            return len(self.image_paths)

    def get_scene_name(self,idx):
        return self.image_paths[idx].split('/')[2]



if __name__ == "__main__":
    dataset=Semantic3dData('data/pointcloud_images_3_2')
    a,p,n=dataset[1]
    a.show(); p.show(); n.show()

