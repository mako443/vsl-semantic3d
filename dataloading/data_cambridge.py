import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import os
import pickle
from scipy.spatial.transform import Rotation

class CambridgeDataset(Dataset):
    def __init__(self, dirpath_main,split, transform=None):
        assert split in ('train','test')
        assert os.path.isdir(dirpath_main)

        self.transform=transform

        self.scene_names=sorted([folder_name for folder_name in os.listdir(dirpath_main) if os.path.isdir(os.path.join(dirpath_main,folder_name))]) #Sorting scene-names
        print('Scene names:',self.scene_names)
        
        self.image_paths=[]
        self.image_positions=[]
        self.image_orientations=[]
        self.image_scene_names=[]

        frames_skipped=0
        for scene_name in self.scene_names:
            file_path=os.path.join(dirpath_main, scene_name,f'dataset_{split}.txt')
            assert os.path.isfile(file_path)
            with open(file_path,'r') as f:
                lines=f.readlines()

            for line in lines:
                if not line.startswith('seq'): continue
                line_split=line.split()
                frame_position=(float(line_split[1]),float(line_split[2]),float(line_split[3]))
                frame_orientation=Rotation.from_quat([float(line_split[4]),float(line_split[5]),float(line_split[6]),float(line_split[7])])

                if np.max(np.abs(frame_position))>1e4: continue # One frame had an invalid position value
                self.image_paths.append( os.path.join(dirpath_main,scene_name,line_split[0]) )
                self.image_positions.append( frame_position )
                self.image_orientations.append(frame_orientation.as_euler('xyz')[0])
                self.image_scene_names.append(scene_name)

        self.image_paths=np.array(self.image_paths)
        self.image_positions=np.array(self.image_positions)
        self.image_orientations=np.array(self.image_orientations)
        self.image_scene_names=np.array(self.image_scene_names)

        print(f'CambridgeDataset(): split <{split}>, {len(self.scene_names)} scenes, {len(self.image_paths)} images, skipped: {frames_skipped}')
        assert len(self.image_paths)==len(self.image_positions)==len(self.image_scene_names)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,index):
        image  =  Image.open(self.image_paths[index]).convert('RGB')     
        if self.transform:
            image=self.transform(image)

        return image


# dataset_train=CambridgeDataset('data_cambridge','train')
# print(np.min(dataset_train.image_orientations), np.max(dataset_train.image_orientations))
# dataset_test =CambridgeDataset('data_cambridge','test')
# print(np.min(dataset_test.image_orientations), np.max(dataset_test.image_orientations))
