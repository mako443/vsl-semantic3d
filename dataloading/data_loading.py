import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import os
import pickle
from semantic.patches import Patch


'''
TODO
-Loader is used for all training&eval, never for data creation!
-Unify all loading to one unified, PyTorch-based Data-Loader
-Anchor-pairs via inheritance
-Load SGs and Texts! (Don't create here!)
'''

#Dataset is used for all loading during all training and evaluation, but never during data creation!
class Semantic3dDataset(Dataset):
    def __init__(self, dirpath_main, transform=None, image_limit=None, split_indices=None):
        assert os.path.isdir(dirpath_main)

        self.dirpath_main=dirpath_main
        self.transform=transform

        #Move to inherited
        #self.positive_thresh=np.array((7.5,2*np.pi/12*1.01))
        #self.negative_thresh=np.array((10,np.pi / 2))

        '''
        Data is structured in directories and unordered dictionaries
        From here on, all folders and image-names will be sorted
        '''
        self.scene_names=sorted([folder_name for folder_name in os.listdir(dirpath_main) if os.path.isdir(os.path.join(dirpath_main,folder_name))]) #Sorting scene-names
        print(f'Semantic3dData with {len(self.scene_names)} total scenes: {self.scene_names}')

        self.image_paths=[]
        self.image_poses=np.array([],np.float32).reshape(0,6)
        self.image_patches=[]
        self.image_scenegraphs=[]
        #Go through all scenes and image-names
        for scene_name in self.scene_names:
            #Load image_paths
            scene_path=os.path.join(dirpath_main,scene_name)
            scene_path_rgb=os.path.join(scene_path, 'rgb')
            scene_image_names= sorted(os.listdir(scene_path_rgb)) #Sorting image-names
            self.image_paths.extend( [os.path.join(scene_path_rgb, image_name) for image_name in scene_image_names] ) 

            #Load poses
            scene_poses_dict=pickle.load( open(os.path.join(scene_path,'poses.pkl'), 'rb') )
            scene_image_poses=np.array([ scene_poses_dict[image_name] for image_name in scene_image_names ])
            self.image_poses=np.vstack(( self.image_poses, scene_image_poses ))

            #Load patches #TODO: make optional depending on data creation approach
            scene_patches_dict=pickle.load( open(os.path.join(scene_path,'patches.pkl'), 'rb') )
            scene_patches= [ scene_patches_dict[image_name] for image_name in scene_image_names ]
            self.image_patches.extend(scene_patches)

            #Load Scene-Graphs
            

        assert len(self.image_paths)==len(self.image_poses)==len(self.image_patches)

        if split_indices is None:
            print('No splitting...')
        else:
            print(f'Splitting, using {np.sum(split_indices)} of {len(self.image_paths)} indices')
            assert len(split_indices)==len(self.image_paths)
            self.image_paths=self.image_paths[split_indices]
            self.image_poses=self.image_poses[split_indices]
            self.image_patches=self.image_patches[split_indices]
            assert len(self.image_paths)==len(self.image_poses)

        if image_limit and image_limit>0:
            self.image_limit=image_limit
        else:
            self.image_limit=None

    def __len__(self):
        if self.image_limit:
            return min(self.image_limit, len(self.image_paths))
        else:
            return len(self.image_paths)

    def get_scene_name(self,idx):
        return self.image_paths[idx].split('/')[2]     

    #Returns the image at the current index
    def __getitem__(self,index):       
        image  =  Image.open(self.image_paths[index]).convert('RGB')     

        if self.transform:
            image = self.transform(image)
            
        return image

#Subclass to load the images as trainig triplets
#Also load SG/text triplets?
class Semantic3dDatasetTriplet(Semantic3dDataset):
    def __init__(self, dirpath_main, transform=None, image_limit=None, split_indices=None):
        super().__init__(dirpath_main, transform, image_limit, split_indices)

    #TODO
    def __getitem__(self, anchor_index):
        pass

if __name__ == "__main__":
    dataset=Semantic3dDatasetTriplet('data/pointcloud_images_3_2_depth')

