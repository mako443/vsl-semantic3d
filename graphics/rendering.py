import numpy as np
import os
#import main
import pickle
from semantic.clustering import ClusteredObject2

class Pose:
    def __init__(self, scene_name, eye, right, up, forward):
        self.scene_name=scene_name
        self.eye=eye
        self.right=right
        self.up=up
        self.forward=forward

CLASSES_DICT={'unlabeled': 0, 'man-made terrain': 1, 'natural terrain': 2, 'high vegetation': 3, 'low vegetation': 4, 'buildings': 5, 'hard scape': 6, 'scanning artefacts': 7, 'cars': 8}
CLASSES_COLORS={'unlabeled': (255,255,255), 'man-made terrain': (60,30,30), 'natural terrain': (30,60,30), 'high vegetation': (120,255,120), 'low vegetation': (80,255,80), 'buildings': (255,255,0), 'hard scape': (0,255,255), 'scanning artefacts': (255,0,0), 'cars': (0,0,255)}

def reduce_points(points, max_points):
    step=int(np.ceil(len(points)/max_points))
    points=points[::step].copy()
    return points

#Expand labelled points into unknown points: Find the nearest neighbors via KD-Tree, label all currently unkown points like their closest known neighbor
#Better than other logic from main.load_files()?
def expand_labels(xyz, rgb, lbl, iterations):
    pass


#Load as rgb, treat label and rgb the same
def load_files2(base_path, scene_name, max_points=int(28e6)):
    p_xyz   =os.path.join(base_path,scene_name+'.xyz.npy')
    p_rgb   =os.path.join(base_path,scene_name+'.rgb.npy')
    p_labels=os.path.join(base_path,scene_name+'.labels.npy')

    assert os.path.isfile(p_xyz) and os.path.isfile(p_rgb) and os.path.isfile(p_labels)

    #Load numpy files
    xyz, rgb, lbl=np.load(open(p_xyz,'rb')),np.load(open(p_rgb,'rb')),np.load(open(p_labels,'rb'))

    #TODO: blunt artifact removal ok?
    #Remove artifacts
    mask= lbl!=CLASSES_DICT['scanning artefacts']
    print(f'Retaining {np.sum(mask) / len(xyz) : 0.3} of points after artifact removal')
    xyz, rgb, lbl=xyz[mask,:], rgb[mask,:], lbl[mask]

    #Reduce the points via stepping to prevent memory erros
    xyz, rgb, lbl=reduce_points(xyz, max_points=max_points), reduce_points(rgb, max_points=max_points), reduce_points(lbl, max_points=max_points)

    return xyz, rgb, lbl


'''
Rendering via 3D clusters
'''
def capture_view(visualizer, pose, scene_objects):
    pass

if __name__ == "__main__":
    scene_name='domfountain_station1_xyz_intensity_rgb'
    scene_objects=pickle.load( open('data/numpy/'+scene_name+'.objects.pkl','rb'))

    #xyz, rgba, labels_rgba=main.load_files('data/numpy/'+scene_name, remove_artifacts=True, remove_unlabeled=True, max_points=int(20e6))



#RENDERING OPTIONS: 
#Open3D SLURM: Can get Pinhole params, but blacked out or otherwise unfeasible?
#PyVista SLURM: Perspective anders (korrekt?), camera also seems accessible: https://github.com/pyvista/pyvista-support/issues/85, not blacked
#-> Beide waren schlecht zum "Umschauen", Perspektiven leicht verschieden...

'''
TODO
-Possibly try SLURM again? -> No ✖, maybe if voxel-downsample fails
-build from main&capturing, remove capturing
-voxel-downsample before rendering? Then on PyVista or Open3D?
'''



'''
Interactive PyVista on SLURM:

srun xvfb-run python3 -i

import numpy as np
import pyvista      

points=np.random.rand(100,3)
mesh=pyvista.PolyData(points)

plotter=pyvista.Plotter(off_screen=True)
_=plotter.add_mesh(mesh)
res=plotter.show(screenshot='pyvista.png', interactive=False)
'''



