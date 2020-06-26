import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import os

class Semantic3dData(Dataset):
    def __init__(self, dirpath_main, transform=None, image_limit=None):
        assert os.path.isdir(dirpath_main)

        self.dirpath_main=dirpath_main
        self.transform=transform

        self.positive_thresh=np.array((2.5,np.pi/3))
        self.negative_thresh=np.array((10,np.pi/2))

        self.scene_names=[folder_name for folder_name in os.listdir(dirpath_main) if os.path.isdir(os.path.join(dirpath_main,folder_name))]
        print('Semantic3dData scene names: ',self.scene_names)

        self.images_dict={}
        #Read all image names
        for scene_name in self.scene_names:
            p=os.path.join(dirpath_main,scene_name,'rgb')
            self.images_dict[scene_name]=sorted(list(os.listdir(p)))

        #Read all poses
        self.poses_dict={}
        for scene_name in self.scene_names:
            p=os.path.join(dirpath_main,scene_name,'poses.npy')
            assert os.path.isfile(p)
            self.poses_dict[scene_name]=np.fromfile(p).reshape((-1,6)) # [x,y,z,phi,theta,r]

        if image_limit and image_limit>0:
            self.image_limit=image_limit

    #Retrieves a triplet with [idx] as anchor, currently all from same scene TODO: chance to take negative from other scene randomly
    def __getitem__(self,idx):
        count=0
        for scene_name in self.scene_names:
            if count+len(self.images_dict[scene_name]) <= idx:
                count+=len(self.images_dict[scene_name])
            else:
                anchor_index=idx-count
                break

        scene_poses=self.poses_dict[scene_name]
        location_dists=scene_poses[:,0:3]-scene_poses[anchor_index,0:3]
        location_dists=np.linalg.norm(location_dists,axis=1)

        orientation_dists=np.abs(scene_poses[:,3]-scene_poses[anchor_index,3]) #CARE: Currently just considering phi
        orientation_dists=np.minimum(orientation_dists,2*np.pi-orientation_dists)

        #Find positive index
        location_dists[anchor_index]=np.inf
        orientation_dists[anchor_index]=np.inf
        indices= (location_dists<self.positive_thresh[0]) & (orientation_dists<self.positive_thresh[1])
        assert np.sum(indices)>0
        indices=np.argwhere(indices==True).flatten()
        positive_index=np.random.choice(indices)

        #Find negative index
        location_dists[anchor_index]=0
        orientation_dists[anchor_index]=0
        indices= (location_dists>=self.negative_thresh[0]) & (orientation_dists>=self.negative_thresh[1])
        assert np.sum(indices)>0
        indices=np.argwhere(indices==True).flatten()
        negative_index=np.random.choice(indices)

        #print(anchor_index, positive_index, negative_index)

        anchor  =  Image.open(os.path.join(self.dirpath_main,scene_name,'rgb',self.images_dict[scene_name][anchor_index])).convert('RGB')
        positive = Image.open(os.path.join(self.dirpath_main,scene_name,'rgb',self.images_dict[scene_name][positive_index])).convert('RGB')
        negative = Image.open(os.path.join(self.dirpath_main,scene_name,'rgb',self.images_dict[scene_name][negative_index])).convert('RGB')
        #pose=self.poses_dict[scene_name][anchor_index]

        #image_rgb=Image.open(path_rgb)

        if self.transform:
            anchor, positive, negative = self.transform(anchor),self.transform(positive),self.transform(negative)

        return anchor,positive, negative

    def __len__(self):
        if self.image_limit:
            return self.image_limit
        else:
            return np.sum([len(self.images_dict[scene_name]) for scene_name in self.scene_names])


if __name__ == "__main__":
    dataset=Semantic3dData('data/pointcloud_images')
    a,p,n=dataset[0]
    a.show(); p.show(); n.show()