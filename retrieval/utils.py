import os
import numpy as np

def get_split_indices(test_split,total_images):
    if test_split is not None:
        test_indices=np.zeros(total_images,dtype=np.bool)
        test_indices[::test_split]=True
        train_indices=np.invert(test_indices)  
        return train_indices, test_indices
    else:
        return None