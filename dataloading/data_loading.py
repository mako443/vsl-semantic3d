import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import os
import pickle
from graphics.imports import Pose
from semantic.patches import Patch
from semantic.imports import SceneGraph, SceneGraphObject, ViewObject


'''
TODO
-Loader is used for all training&eval, never for data creation! ✓
-Unify all loading to one unified, PyTorch-based Data-Loader ✓
-Anchor-pairs via inheritance ✓
-Load SGs and Texts! (Don't create here!) ✓


'''

#Dataset is used for all loading during all training and evaluation, but never during data creation!
class Semantic3dDataset(Dataset):
    def __init__(self, dirpath_main, transform=None, image_limit=None, split_indices=None, load_viewObjects=True, load_sceneGraphs=True):
        assert os.path.isdir(dirpath_main)

        self.dirpath_main=dirpath_main
        self.transform=transform

        '''
        Data is structured in directories and unordered dictionaries
        From here on, all folders and image-names will be sorted
        '''
        self.scene_names=sorted([folder_name for folder_name in os.listdir(dirpath_main) if os.path.isdir(os.path.join(dirpath_main,folder_name))]) #Sorting scene-names

        #self.scene_names=('sg27_station5_intensity_rgb',)
        #print('CARE / DEBUG ONE SCENE')
        print(f'Semantic3dData with {len(self.scene_names)} total scenes: {self.scene_names}')

        self.image_paths,self.image_poses,self.image_positions,self.image_orientations,self.image_scene_names,self.view_objects,self.view_scenegraphs=[],[],[],[],[],[],[]
        #TODO: image_captions

        #Go through all scenes and image-names
        for scene_name in self.scene_names:
            #Load image_paths
            scene_path=os.path.join(dirpath_main,scene_name)
            scene_path_rgb=os.path.join(scene_path, 'rgb')
            scene_image_names= sorted(os.listdir(scene_path_rgb)) #Sorting image-names
            self.image_paths.extend( [os.path.join(scene_path_rgb, image_name) for image_name in scene_image_names] ) 

            #Load poses
            #TODO: convert poses back to array
            scene_poses_dict=pickle.load( open(os.path.join(scene_path,'poses_rendered.pkl'), 'rb') ) #CARE: poses or poses rendered?
            scene_poses=[scene_poses_dict[image_name] for image_name in scene_image_names]
            self.image_poses.extend(scene_poses)
            self.image_positions.extend( [pose.eye for pose in scene_poses] )
            self.image_orientations.extend( [pose.phi for pose in scene_poses] )
            self.image_scene_names.extend( [scene_name for i in range(len(scene_image_names))] )

            # #Load patches #TODO: make optional depending on data creation approach
            # scene_patches_dict=pickle.load( open(os.path.join(scene_path,'patches.pkl'), 'rb') )
            # scene_patches= [ scene_patches_dict[image_name] for image_name in scene_image_names ]
            # self.image_patches.extend(scene_patches)

            #Load View-Objects
            if load_viewObjects:
                scene_viewobjects_dict=pickle.load( open(os.path.join(scene_path,'view_objects.pkl'), 'rb') )
                scene_viewobjects= [scene_viewobjects_dict[image_name] for image_name in scene_image_names]
                self.view_objects.extend(scene_viewobjects)

            #Load Scene-Graphs
            if load_sceneGraphs:
                scene_scenegraphs_dict=pickle.load( open(os.path.join(scene_path,'scene_graphs.pkl'), 'rb') )
                scene_scenegraphs= [ scene_scenegraphs_dict[image_name] for image_name in scene_image_names ]
                self.view_scenegraphs.extend(scene_scenegraphs)

        self.image_paths,self.image_poses,self.image_positions,self.image_orientations,self.view_objects,self.view_scenegraphs=np.array(self.image_paths),np.array(self.image_poses),np.array(self.image_positions),np.array(self.image_orientations),np.array(self.view_objects),np.array(self.view_scenegraphs)
        
        assert self.image_positions.shape[1]==3
        assert len(self.image_paths)==len(self.image_poses) #==len(self.view_objects)==len(self.view_scenegraphs)

        if split_indices is None:
            print('No splitting...')
        else:
            print(f'Splitting, using {np.sum(split_indices)} of {len(self.image_paths)} indices')
            assert len(split_indices)==len(self.image_paths)
            self.image_paths=self.image_paths[split_indices]
            self.image_poses=self.image_poses[split_indices]
            self.image_positions=self.image_positions[split_indices]
            self.image_orientations=self.image_orientations[split_indices]
            if load_viewObjects: self.view_objects=self.view_objects[split_indices]
            if load_sceneGraphs: self.view_scenegraphs=self.view_scenegraphs[split_indices]
            assert len(self.image_paths)==len(self.image_poses)

        if image_limit and image_limit>0:
            self.image_limit=image_limit
        else:
            self.image_limit=None
        print()

    def __len__(self):
        if self.image_limit:
            return min(self.image_limit, len(self.image_paths))
        else:
            return len(self.image_paths)

    # def get_scene_name(self,idx):
    #     return self.image_paths[idx].split('/')[2]     

    #Returns the image at the current index
    def __getitem__(self,index):       
        image  =  Image.open(self.image_paths[index]).convert('RGB')     

        if self.transform:
            image = self.transform(image)
            
        return image

#Subclass to load the images as trainig triplets
#Also load SG/text triplets?
class Semantic3dDatasetTriplet(Semantic3dDataset):
    def __init__(self, dirpath_main, transform=None, image_limit=None, split_indices=None, load_viewObjects=True, load_sceneGraphs=True):
        super().__init__(dirpath_main, transform, image_limit, split_indices, load_viewObjects, load_sceneGraphs)
        self.positive_thresh=(7.5, 2*np.pi/10*1.01) #The 2 images left&right
        self.negative_thresh=(10,  np.pi / 2)

    def __getitem__(self, anchor_index):
        scene_name=self.image_scene_names[anchor_index]

        pos_dists=np.linalg.norm(self.image_positions[:]-self.image_positions[anchor_index], axis=1)
        ori_dists=np.abs(self.image_orientations[:]-self.image_orientations[anchor_index])
        ori_dists=np.minimum(ori_dists, 2*np.pi-ori_dists)        

        #Find positive index
        pos_dists[anchor_index]=np.inf
        ori_dists[anchor_index]=np.inf
        indices= (pos_dists<self.positive_thresh[0]) & (ori_dists<self.positive_thresh[1]) & (np.core.defchararray.find(self.image_paths, scene_name)!=-1) #location&ori. dists small enough, same scene
        #assert np.sum(indices)>0
        if not np.sum(indices)>0:
            #print(f'No positive indices for anchor {anchor_index}, using image itself') #TODO: investigate
            #indices[anchor_index]=True
            print(f'No positive indices for anchor {anchor_index}, using left or right') #TODO: investigate
            if anchor_index>0: indices[ anchor_index-1 ]=True
            if anchor_index<len(self)-1: indices[anchor_index+1]=True

        indices=np.argwhere(indices==True).flatten()
        #if len(indices)<2: print('Warning: only 1 pos. index choice')
        positive_index=np.random.choice(indices) #OPTION: positive index criterion | care: "always to same side" otherwise

        #OPTION: Negative selection
        if np.random.choice([True,False]): #Pick an index of the same scene
            pos_dists[anchor_index]=0.0
            ori_dists[anchor_index]=0.0
            indices= (pos_dists>=self.negative_thresh[0]) & (ori_dists>=self.negative_thresh[1]) & (np.core.defchararray.find(self.image_paths, scene_name)!=-1)# loc.&ori. dists big enough, same scene
            assert np.sum(indices)>0
            indices=np.argwhere(indices==True).flatten()
            negative_index=np.random.choice(indices)            
        else: #Pick from other scene
            indices=np.flatnonzero(np.core.defchararray.find(self.image_paths, scene_name)==-1)
            negative_index=np.random.choice(indices)

        anchor  =  Image.open(self.image_paths[anchor_index]).convert('RGB')
        positive = Image.open(self.image_paths[positive_index]).convert('RGB')
        negative = Image.open(self.image_paths[negative_index]).convert('RGB')    

        if self.transform:
            anchor, positive, negative = self.transform(anchor),self.transform(positive),self.transform(negative)

        return anchor, positive, negative

if __name__ == "__main__":
    dataset=Semantic3dDatasetTriplet('data/pointcloud_images_o3d_merged_1scene', load_viewObjects=False, load_sceneGraphs=False)

