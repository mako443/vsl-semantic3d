import numpy as np
import cv2
import os
from main import load_files, classes_dict
import open3d
import pptk
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import pickle
import semantic.utils

class ClusteredObject:
    def __init__(self, label, bbox, num_points):
        self.label=label
        self.bbox=bbox #BBOX as [xmin,ymin,zmin,xmax,ymax,zmax]
        self.num_points=num_points
        self.bbox_projected=None

    def __str__(self):
        return f'ClusteredObject: {self.label} at {self.bbox}, {self.num_points} points'

    #Naively project all 8 bbox-points -> formulate as new bbox
    #CARE: bbox_image now in image-coordinates!
    def project(self,I,E):
        bbox_points=np.array(np.meshgrid([ self.bbox[0],self.bbox[3] ],[ self.bbox[1],self.bbox[4] ],[ self.bbox[2],self.bbox[5] ])).T.reshape((8,3))
        bbox_points= [semantic.utils.project_point(I,E, point) for point in bbox_points]
        self.bbox_projected=np.concatenate((np.min(bbox_points, axis=0), np.max(bbox_points, axis=0)), axis=None)
    
    def draw_rectangle(self, img):
        if self.bbox_projected is None:
            print("ClusteredObject::draw_rectangle(): not projected")
        else:
            b=np.int32(self.bbox_projected)
            _=cv2.rectangle(img,(b[0], b[1]), (b[3],b[4]), (255,255,255))
        


#TODO: remove scanning artifacts by clustering -> reject all in BBOX?
#Assume pre-process: Full Voxel-Downsampling -> reduce to 100k 
CLUSTERING_OPTIONS={
    'man-made terrain': {'max_points':60000, 'eps': 1.5, 'min_samples': 300},
    'natural terrain': {'max_points':30000, 'eps': 1.5, 'min_samples': 75},
    'high vegetation': {'max_points':60000, 'eps': 1.5, 'min_samples': 150},
    'low vegetation': {'max_points':30000, 'eps': 1.0, 'min_samples': 150}, #60k crashed for domfountain_station1
    'buildings': {'max_points':30000, 'eps': 4.0, 'min_samples': 300},
    'hard scape': {'max_points':30000, 'eps': 1.75, 'min_samples': 300},
    'cars': {'max_points':60000, 'eps': 0.75, 'min_samples': 300},
}    

def load_files_for_label(base_path,label, max_points=int(60e6)):
    p_xyz=base_path+'.xyz.npy'
    p_rgb=base_path+'.rgb.npy'
    p_labels=base_path+'.labels.npy'

    assert os.path.isfile(p_xyz) and os.path.isfile(p_rgb) and os.path.isfile(p_labels)

    xyz, lbl=np.load(open(p_xyz,'rb')),np.load(open(p_labels,'rb'))

    xyz=xyz[lbl==label,:].copy()
    print(f'Found {len(xyz)} points for label {label}')

    if len(xyz)<1000: #Not enough points for that label
        return None

    #Limit points
    if max_points is not None:
        step=int(np.ceil(len(xyz)/max_points))
        xyz= xyz[::step,:].copy()
        print(f'Limited points, step {step}, num-points: {len(xyz)}')    

    return xyz

#New version for dbscan, other faulty?!
def visualize_dbscan(xyz,labels):
    print(f'label min {np.min(labels)} max {np.max(labels)}')
    rgb=np.zeros_like(xyz)
    for label_value in np.unique(labels):
        if label_value==-1:
            color=(1,0,0) #Red marks unmarked!!
        else:
            color=np.random.rand(3)
        rgb[labels==label_value,:]=color
    viewer=pptk.viewer(xyz)
    viewer.attributes(rgb)
    viewer.set(point_size=0.02)
    return viewer

def cluster_scene(scene_name, return_visualization=False):
    scene_objects=[]
    vis_xyz=np.array([]).reshape((0,3))
    vis_rgb=np.array([]).reshape((0,3))

    for label in CLUSTERING_OPTIONS.keys():
    #for label in ('cars',):
        options=CLUSTERING_OPTIONS[label]

        #Load all points of that label w/o reduction
        xyz=load_files_for_label('data/numpy/'+scene_name, label=classes_dict[label], max_points=None)
        if xyz is None:
            print(f'No points for label {label} in {scene_name}, skipping')
            continue

        #Downsample via Voxel-Grid to remove dense clusters
        pcd=open3d.geometry.PointCloud()
        pcd.points=open3d.utility.Vector3dVector(xyz)
        down_pcd=pcd.voxel_down_sample(voxel_size=0.02)
        xyz=np.asarray(down_pcd.points)     

        #Reduce the points further through simple removal to 100k points
        xyz=semantic.utils.reduce_points(xyz, int(1e5))           

        #Run DB-Scan to find the actual clusters
        cluster=DBSCAN(eps=options['eps'], min_samples=options['min_samples'], leaf_size=30, n_jobs=-1).fit(xyz)

        print(f'Clustering {scene_name}, label <{label}>: {np.max(cluster.labels_)+1} objects')

        for label_value in range(0, np.max(cluster.labels_)+1):
            object_xyz=xyz[cluster.labels_ == label_value]
            bbox=np.concatenate((np.min(object_xyz, axis=0), np.max(object_xyz, axis=0)), axis=None)
            scene_objects.append(ClusteredObject(label, bbox, len(object_xyz)))

            if return_visualization:
                object_rgb=np.random.rand(3)*np.ones_like(object_xyz)
                vis_xyz, vis_rgb=np.vstack((vis_xyz,object_xyz)), np.vstack((vis_rgb,object_rgb))

    print(f'==> Clustered {scene_name}, {len(scene_objects)} objects total')
    if return_visualization:
        return scene_objects, vis_xyz, vis_rgb
    else:
        return scene_objects

'''
METHODS in sklearn:
Spectral: need #clusters
Ward: RAM-limit 30k points, runs, promising / minor problems with few points&merging objects, sometimes objects still split
DBSCAN: crashes 
OPTICS: slow
Birch: can work with #cluster estimator? -> Apparently not helpful ✖
SLURM Ward, DBSCAN; grid-segment the scene also? -> too slow / too much mem consumption, objects still split
'''
def cluster_pointcloud():    
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=150.0).fit(xyz)
    labels=clustering.labels_

'''
TODO
-Always use Voxel-Downsampling? Load all -> downsample with all -> reduce -> scan 
-Do all scenes, visualize all objects reduced points (mixed labels)

-Pull big scenes with 100M
'''
if __name__ == "__main__":
    scene_name='domfountain_station1_xyz_intensity_rgb'
    scene_objects, xyz, rgb=cluster_scene(scene_name, return_visualization=True)
    #xyz, rgba, labels_rgba=load_files('data/numpy/'+scene_name,max_points=int(10e6))
    #scene_objects=pickle.load(open('data/numpy/'+scene_name+'.objects.pkl','rb'))

    v=pptk.viewer(xyz)
    v.attributes(rgb)
    v.set(point_size=0.025)

    bushes= [ o for o in scene_objects if "high" in o.label]



    quit()

    for scene_name in ('domfountain_station1_xyz_intensity_rgb','sg27_station2_intensity_rgb','untermaederbrunnen_station1_xyz_intensity_rgb','neugasse_station1_xyz_intensity_rgb'):
        print()
        print("Scene: ",scene_name)
        scene_objects=cluster_scene(scene_name)

        print('Saving scene objects...', len(scene_objects),'objects in total')
        pickle.dump( scene_objects, open('data/numpy/'+scene_name+'.objects.pkl', 'wb'))

    quit()

    scene_name='domfountain_station1_xyz_intensity_rgb'
    xyz=load_files_for_label('data/numpy/'+scene_name, label=classes_dict['cars'], max_points=None)

    #Downsample xyz via Voxel-Grid
    pcd=open3d.geometry.PointCloud()
    pcd.points=open3d.utility.Vector3dVector(xyz)
    down_pcd=pcd.voxel_down_sample(voxel_size=0.02)
    xyz=np.asarray(down_pcd.points)
    print('After voxel:',len(xyz))

    #Reduce further through simple removal
    xyz=semantic.utils.reduce_points(xyz, int(1e5))

    print(xyz.shape)
    clustering=DBSCAN(eps=1.0, min_samples=300, leaf_size=30, n_jobs=-1).fit(xyz)
    visualize_dbscan(xyz, clustering.labels_)

    quit()

    #xyz=load_files_for_label('data/numpy/'+scene_name, label=classes_dict['high vegetation'], max_points=int(50e3))

    #150 min_samples for 30k
    #Cars, 60k with 300 samples
    #clustering=DBSCAN(eps=1.0, min_samples=300, leaf_size=30, n_jobs=-1).fit(xyz) #Runs for 60k, seems good for cars, leaf doesn't change much, more points might help <-> adjust min_samples

    #Buildings, 30k, seem to come out as walls! ✓
    #clustering=DBSCAN(eps=4.0, min_samples=150, leaf_size=30, n_jobs=-1).fit(xyz)

    #Low veg, 60k
    #clustering=DBSCAN(eps=1.0, min_samples=150, leaf_size=30, n_jobs=-1).fit(xyz)

    #Hard scape, 30k, seems ok
    #clustering=DBSCAN(eps=1.0, min_samples=150, leaf_size=30, n_jobs=-1).fit(xyz)

    #Nat terrain, 30k max anyhow, CARE: scenes w/ more points
    #clustering=DBSCAN(eps=1.5, min_samples=75, leaf_size=30, n_jobs=-1).fit(xyz)

    #man-made terrain, 50k, care: BBox here ok?
    #clustering=DBSCAN(eps=1.5, min_samples=300, leaf_size=30, n_jobs=-1).fit(xyz)

    #High veg, 60k
    #clustering=DBSCAN(eps=1.0, min_samples=150, leaf_size=30, n_jobs=-1).fit(xyz)

    #print(clustering.labels_.max())
    
    #viewer=visualize_dbscan(xyz,clustering.labels_)

    quit()
