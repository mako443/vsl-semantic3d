import numpy as np
import cv2
import os
#from main import load_files, CLASSES_DICT
import open3d
import pptk
from sklearn.cluster import DBSCAN
import pickle
from .geometry import project_point, IMAGE_WIDHT, IMAGE_HEIGHT
#from graphics.rendering import Pose, CLASSES_DICT, CLASSES_COLORS
import semantic.utils

class ClusteredObject:
    #TODO: input as floor_rect, (zmin,zmax), in project -> cv2.boxPoints -> meshgrid -> project
    def __init__(self, label, bbox, num_points):
        self.label=label
        self.bbox=bbox #BBOX as [xmin,ymin,zmin,xmax,ymax,zmax]
        self.num_points=num_points
        self.bbox_projected=None
        self.image_rect=None

    def __str__(self):
        return f'ClusteredObject: {self.label} at {self.bbox}, {self.num_points} points'

    def get_bbox_points(self):
        return np.array(np.meshgrid([ self.bbox[0],self.bbox[3] ],[ self.bbox[1],self.bbox[4] ],[ self.bbox[2],self.bbox[5] ])).T.reshape((8,3))

    def get_bbox_projected_points(self):
        return np.array(np.meshgrid([ self.bbox_projected[0],self.bbox_projected[3] ],[ self.bbox_projected[1],self.bbox_projected[4] ],[ self.bbox_projected[2],self.bbox_projected[5] ])).T.reshape((8,3))        

    #Naively project all 8 bbox-points -> formulate as new bbox
    #CARE: bbox_image now in image-coordinates!
    #TODO: project like todo above, set floor-box-points and image-box-points
    def project(self,I,E):
        bbox_points=self.get_bbox_points()
        bbox_points_projected= np.array([project_point(I,E, point) for point in bbox_points])
        self.bbox_projected=np.concatenate((np.min(bbox_points_projected, axis=0), np.max(bbox_points_projected, axis=0)), axis=None)
        self.image_rect=cv2.minAreaRect(bbox_points_projected[:,0:2].astype(np.float32)) #Bug w/o float32
    
    #Does not check occlusions!
    def in_fov(self):
        if self.bbox_projected is None:
            print("ClusteredObject::is_visible(): not projected")        
        else:
            return self.bbox_projected[2]>0 and (self.bbox_projected[3]>0.1*IMAGE_WIDHT or self.bbox_projected[0]<0.9*IMAGE_WIDHT) and (self.bbox_projected[4]>0.1*IMAGE_HEIGHT or self.bbox_projected[2]<0.9*IMAGE_HEIGHT)
    
    def draw_on_image(self, img):
        if self.bbox_projected is None:
            print("ClusteredObject::draw_rectangle(): not projected")
        else:
            b=np.int32(self.bbox_projected)
            #_=cv2.rectangle(img,(b[0], b[1]), (b[3],b[4]), (255,255,0),thickness=2)
            box=np.int0(cv2.boxPoints(self.image_rect))
            cv2.drawContours(img,[box],0,(255,255,0),thickness=2)
            _=cv2.putText(img,self.label,(b[0]+5, b[1]+5),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0))

    # def get_image_rect(self):
    #     points=self.get_bbox_projected_points()
    #     return cv2.minAreaRect(points[:,0:2].astype(np.float32)) #Bug w/o float32

class ClusteredObject2:
    def __init__(self, label, bbox_points, num_points):
        self.label=label
        self.bbox_points_w=bbox_points # BBox in world-coords, 8x 3d-points
        self.bbox_points_i=None
        self.bbox_rect_i=None # BBox in image-coords, available after self.project(), cv2.RotatedRect object
        self.mindist_i, self.maxdist_i= None, None #Distance in image-coords, available after self.project
        self.num_points=num_points

    def __str__(self):
        return f'ClusteredObject2: {self.label} at {np.mean(self.bbox_points_w, axis=0)}, {self.num_points} points'

    def project(self, I,E):
        #Project all world-coordinate points to 3d image-points
        self.bbox_points_i = np.array( [project_point(I,E, point) for point in self.bbox_points_w] )
        #Set bbox_i as rotated rect in image-plane
        self.bbox_rect_i=cv2.minAreaRect(self.bbox_points_i[:,0:2].astype(np.float32)) #Bug w/o float32
        #Set image-plane distances
        self.mindist_i, self.maxdist_i = np.min(self.bbox_points_i[:,2]), np.max(self.bbox_points_i[:,2])

    # is_in_fov() and is_occluded() both external

    def draw_on_image(self,img):
        assert self.bbox_points_i is not None
        box=np.int0(cv2.boxPoints(self.bbox_rect_i))
        cv2.drawContours(img,[box],0,(255,255,0),thickness=2)
        #_=cv2.putText(img, self.label, (points[0,0]+5, points[0,1]+5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0))


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

    for label in ('man-made terrain','natural terrain','high vegetation','low vegetation','buildings','hard scape','cars'): #Disregard unknowns and artifacts
    #for label in ('cars',):
        options=CLUSTERING_OPTIONS[label]

        #Load all points of that label w/o reduction
        xyz=load_files_for_label('data/numpy/'+scene_name, label=CLASSES_DICT[label], max_points=None)
        if xyz is None:
            print(f'No points for label {label} in {scene_name}, skipping')
            continue

        #Downsample via Voxel-Grid to remove dense clusters
        pcd=open3d.geometry.PointCloud()
        pcd.points=open3d.utility.Vector3dVector(xyz)
        down_pcd=pcd.voxel_down_sample(voxel_size=0.02)
        xyz=np.asarray(down_pcd.points)     

        #Reduce the points further through simple stepping to 100k points
        xyz=semantic.utils.reduce_points(xyz, int(1e5))           

        #Run DB-Scan to find the actual clusters
        cluster=DBSCAN(eps=options['eps'], min_samples=options['min_samples'], leaf_size=30, n_jobs=-1).fit(xyz)

        print(f'Clustering {scene_name}, label <{label}>: {np.max(cluster.labels_)+1} objects')

        for label_value in range(0, np.max(cluster.labels_)+1):
            object_xyz=xyz[cluster.labels_ == label_value]
            #bbox=np.concatenate((np.min(object_xyz, axis=0), np.max(object_xyz, axis=0)), axis=None)
            #scene_objects.append(ClusteredObject(label, bbox, len(object_xyz)))

            oriented_bbox=open3d.geometry.OrientedBoundingBox.create_from_points(open3d.utility.Vector3dVector(object_xyz))
            bbox_points=np.asarray( oriented_bbox.get_box_points() )
            scene_objects.append( ClusteredObject2(label, bbox_points, len(object_xyz)) )

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
    # scene_name='domfountain_station1_xyz_intensity_rgb'
    # scene_objects, xyz, rgb=cluster_scene(scene_name, return_visualization=True)
    # #xyz, rgba, labels_rgba=load_files('data/numpy/'+scene_name,max_points=int(10e6))
    # #scene_objects=pickle.load(open('data/numpy/'+scene_name+'.objects.pkl','rb'))

    # v=pptk.viewer(xyz)
    # v.attributes(rgb)
    # v.set(point_size=0.025)

    # #bushes= [ o for o in scene_objects if "high" in o.label]

    # quit()

    '''
    Data creation: Clustered objects
    '''
    for scene_name in ('domfountain_station1_xyz_intensity_rgb','sg27_station2_intensity_rgb','untermaederbrunnen_station1_xyz_intensity_rgb','neugasse_station1_xyz_intensity_rgb'):
        print()
        print("Scene: ",scene_name)
        scene_objects=cluster_scene(scene_name)

        print('Saving scene objects...', len(scene_objects),'objects in total')
        pickle.dump( scene_objects, open('data/numpy/'+scene_name+'.objects.pkl', 'wb'))
        break

    quit()  
