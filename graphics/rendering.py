import numpy as np

class Pose:
    def __init__(self, scene_name, eye, right, up, forward):
        self.scene_name=scene_name
        self.eye=eye
        self.right=right
        self.up=up
        self.forward=forward

CLASSES_DICT={'unlabeled': 0, 'man-made terrain': 1, 'natural terrain': 2, 'high vegetation': 3, 'low vegetation': 4, 'buildings': 5, 'hard scape': 6, 'scanning artefacts': 7, 'cars': 8}
CLASSES_COLORS={'unlabeled': (255,255,255), 'man-made terrain': (60,30,30), 'natural terrain': (30,60,30), 'high vegetation': (120,255,120), 'low vegetation': (80,255,80), 'buildings': (255,255,0), 'hard scape': (0,255,255), 'scanning artefacts': (255,0,0), 'cars': (0,0,255)}



#RENDERING OPTIONS: 
#Open3D SLURM: Can get Pinhole params, but blacked out or otherwise unfeasible?
#PyVista SLURM: Perspective anders (korrekt?), camera also seems accessible: https://github.com/pyvista/pyvista-support/issues/85
#-> Beide waren schlecht zum "Umschauen", Perspektiven leicht verschieden...

'''
TODO
-Possibly try SLURM again? -> No ✖, maybe if voxel-downsample fails
-build from main&capturing
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
plotter.show(screenshot='pyvista.png', interactive=False)
'''



