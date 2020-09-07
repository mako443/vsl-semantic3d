import numpy as np
import cv2
import os
#from main import load_files, CLASSES_DICT
import open3d
import pptk
from sklearn.cluster import DBSCAN
import pickle
#from .geometry import project_point, IMAGE_WIDHT, IMAGE_HEIGHT
from graphics.imports import IMAGE_WIDHT, IMAGE_HEIGHT, CLASSES_DICT, CLASSES_COLORS, COMBINED_SCENE_NAMES
#from graphics.rendering import Pose, CLASSES_DICT, CLASSES_COLORS
import semantic.utils
from .imports import ClusteredObject

#TODO: remove scanning artifacts by clustering -> reject all in BBOX? OR Open3D outlier removal
#CARE: Hard-scape in Bildstein 1?!
#Assume pre-process: Full Voxel-Downsampling -> reduce to 100k 
CLUSTERING_OPTIONS={
    'man-made terrain': {'min_points':500, 'eps': 1.5, 'min_samples': 300},
    'natural terrain': {'min_points':250, 'eps': 1.5, 'min_samples': 75},
    'high vegetation': {'min_points':250, 'eps': 1.5, 'min_samples': 150},
    'low vegetation': {'min_points':125, 'eps': 1.0, 'min_samples': 150},
    'buildings': {'min_points':1000, 'eps': 3.2, 'min_samples': 300},
    'hard scape': {'min_points':250, 'eps': 1.75, 'min_samples': 300},
    'cars': {'min_points':500, 'eps': 0.75, 'min_samples': 300},
}    

def load_files_for_label(base_path,label, max_points=int(10e6)):
    p_xyz=base_path+'.xyz.npy'
    p_rgb=base_path+'.rgb.npy'
    p_labels=base_path+'.lbl.npy'

    assert os.path.isfile(p_xyz) and os.path.isfile(p_rgb) and os.path.isfile(p_labels)

    xyz,rgb,lbl=np.load(open(p_xyz,'rb')),np.load(open(p_rgb,'rb')),np.load(open(p_labels,'rb'))
    rgb=rgb/255.0 #Colors as [0,1]

    xyz=xyz[lbl==label,:].copy()
    rgb=rgb[lbl==label,:].copy()
    print(f'Found {len(xyz)} points for label {label}')

    if len(xyz)<1000: #Not enough points for that label
        return None, None

    #Limit points
    if max_points is not None:
        step=int(np.ceil(len(xyz)/max_points))
        xyz= xyz[::step,:].copy()
        rgb= rgb[::step,:].copy()
        print(f'Limited points, step {step}, num-points: {len(xyz)}')    

    return xyz,rgb

#New version for dbscan, other faulty?!
def visualize_dbscan_pptk(xyz,labels):
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

def visualize_dbscan_o3d(xyz,labels):
    print(f'label min {np.min(labels)} max {np.max(labels)}')
    rgb=np.zeros_like(xyz)
    for label_value in np.unique(labels):
        if label_value==-1:
            color=(1,0,0) #Red marks unmarked!!
        else:
            color=np.random.rand(3)
        rgb[labels==label_value,:]=color

    point_cloud=open3d.geometry.PointCloud()
    point_cloud.points=open3d.utility.Vector3dVector(xyz)    
    point_cloud.colors=open3d.utility.Vector3dVector(rgb)   

    vis = open3d.visualization.Visualizer()
    vis.create_window(width=1620, height=1080)    
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    vis.add_geometry(point_cloud) 
    vis.run()
    return vis

def get_hull_points(points_w):
    assert points_w.shape[1]==3
    pcd=open3d.geometry.PointCloud()
    pcd.points=open3d.utility.Vector3dVector(points_w)
    hull,_=pcd.compute_convex_hull()
    hull_points=np.asarray(hull.vertices)
    return hull_points

def cluster_scene(scene_name, return_visualization=False):
    scene_objects=[]
    vis_xyz=np.array([]).reshape((0,3))
    vis_rgb=np.array([]).reshape((0,3))

    for label in ('man-made terrain','natural terrain','high vegetation','low vegetation','buildings','hard scape','cars'): #Disregard unknowns and artifacts
    #for label in ('buildings',):
        options=CLUSTERING_OPTIONS[label]

        #Load all points of that label w/o reduction
        #xyz, rgb=load_files_for_label('data/numpy/'+scene_name, label=CLASSES_DICT[label], max_points=None)
        xyz, rgb=load_files_for_label('data/numpy_merged/'+scene_name, label=CLASSES_DICT[label], max_points=None)
        if xyz is None:
            print(f'No points for label {label} in {scene_name}, skipping')
            continue

        #Downsample via Voxel-Grid to remove dense clusters
        pcd=open3d.geometry.PointCloud()
        pcd.points=open3d.utility.Vector3dVector(xyz)
        pcd.colors=open3d.utility.Vector3dVector(rgb)
        down_pcd=pcd.voxel_down_sample(voxel_size=0.02)
        xyz=np.asarray(down_pcd.points)     
        rgb=np.asarray(down_pcd.colors)     

        #Reduce the points further through simple stepping to 100k points
        xyz=semantic.utils.reduce_points(xyz, int(1e5))           
        rgb=semantic.utils.reduce_points(rgb, int(1e5))   

        #Run DB-Scan to find the actual clusters
        cluster=DBSCAN(eps=options['eps'], min_samples=options['min_samples'], leaf_size=30, n_jobs=-1).fit(xyz)

        print(f'Clustering {scene_name}, label <{label}>: {np.max(cluster.labels_)+1} objects')

        for label_value in range(0, np.max(cluster.labels_)+1):
            if np.sum( cluster.labels_==label_value ) < options['min_points']:
                print('skipped obj for label (not enough points)',label)
                continue

            object_xyz=xyz[cluster.labels_ == label_value]
            object_rgb=rgb[cluster.labels_ == label_value]
            object_color=np.mean(object_rgb, axis=0)
            #print('col', object_color)
            #hull_points=get_hull_points(object_xyz) #Hull points is unstable after projection / have to clip!
            #bbox=np.concatenate((np.min(object_xyz, axis=0), np.max(object_xyz, axis=0)), axis=None)
            #scene_objects.append(ClusteredObject(label, bbox, len(object_xyz)))

            #oriented_bbox=open3d.geometry.OrientedBoundingBox.create_from_points(open3d.utility.Vector3dVector(object_xyz))
            #bbox_points=np.asarray( oriented_bbox.get_box_points() )
            #scene_objects.append( ClusteredObject2(label, bbox_points, len(object_xyz)) )

            save_xyz=semantic.utils.reduce_points(object_xyz, int(1e4)) #Save max. 10k points per object
            obj=ClusteredObject(scene_name, label, save_xyz, len(object_xyz), object_color)
            scene_objects.append(obj)

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

'''
TODO
-Always use Voxel-Downsampling? Load all -> downsample with all -> reduce -> scan 
-Do all scenes, visualize all objects reduced points (mixed labels)
-Use rotated bboxes after projection!

-Pull big scenes with 100M
-Unstable projection error: Chance (not always) when some points are out of FoV, cv2.projectPoints same&no z -> use mine, norm-restriction didn't work, FoV restriction works
'''
if __name__ == "__main__":
    # scene_name='sg27_station2_intensity_rgb'
    # #label='buildings'
    # scene_objects, vis_xyz, vis_rgb=cluster_scene(scene_name, return_visualization=True)
    # v=pptk.viewer(vis_xyz)
    # v.attributes(vis_rgb)
    # v.set(point_size=0.025)
    # quit()

    #CARE WHAT HAS BEEN RE-CREATED AND WHAT HASN'T!
    '''
    Data creation: Clustered objects
    '''
    for scene_name in COMBINED_SCENE_NAMES:
    #for scene_name in ('sg27_station2_intensity_rgb',):
        print()
        print("Scene: ",scene_name)
        scene_objects=cluster_scene(scene_name)

        print('Saving scene objects...', len(scene_objects),'objects in total')
        pickle.dump( scene_objects, open('data/numpy_merged/'+scene_name+'.objects.pkl', 'wb'))

    quit()      

    ### Project and BBox verify
    # xyz=np.random.rand(100,3)
    # pcd=open3d.geometry.PointCloud()
    # pcd.points=open3d.utility.Vector3dVector(xyz)
    # pcd.colors=open3d.utility.Vector3dVector(np.ones_like(xyz))

    # vis = open3d.visualization.Visualizer()
    # vis.create_window(width=1620, height=1080)    
    # vis.get_render_option().background_color = np.asarray([0, 0, 0])
    # vis.add_geometry(pcd) 
    # vis.run()    

    # control=vis.get_view_control()
    # params=control.convert_to_pinhole_camera_parameters()
    # I,E=params.intrinsic.intrinsic_matrix, params.extrinsic

    # img=np.asarray(vis.capture_screen_float_buffer())
    # #points_i=[ project_point(I,E, point) for point in xyz ]

    # rvec,_=cv2.Rodrigues(E[0:3,0:3])
    # tvec=1*E[0:3,3]
    # print(E)
    # print(tvec)
    # points_i,_=cv2.projectPoints(xyz,rvec,tvec,I,None)
    # points_i=points_i.reshape((-1,2))

    # # quit()

    # for p in points_i:
    #     _=cv2.circle(img, (int(p[0]), int(p[1])), 6, (0,255,0))

    # points_i=np.array(points_i)

    # bbox=np.int0( (np.min(points_i[:,0]), np.min(points_i[:,1]), np.max(points_i[:,0]), np.max(points_i[:,1])) )
    # _=cv2.rectangle( img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,255,0))

    # cv2.imshow("",img); cv2.waitKey()
    # quit()
    ### Project and BBox verify

    ###New verify as above: raw class points -> clustered
    # xyz=load_files_for_label('data/numpy/'+scene_name, label=CLASSES_DICT[label], max_points=None)
    # pcd=open3d.geometry.PointCloud()
    # pcd.points=open3d.utility.Vector3dVector(xyz)
    # pcd.colors=open3d.utility.Vector3dVector(np.ones_like(xyz))

    # #Downsample, set new xyz
    # down_pcd=pcd.voxel_down_sample(voxel_size=0.02)
    # xyz=np.asarray(down_pcd.points)
    # xyz=semantic.utils.reduce_points(xyz, int(1e5))

    # #Cluster, set new xyz
    # options=CLUSTERING_OPTIONS[label]
    # cluster=DBSCAN(eps=options['eps'], min_samples=options['min_samples'], leaf_size=30, n_jobs=-1).fit(xyz)
    # xyz=xyz[cluster.labels_==2,:]
    # pickle.dump(xyz, open("cluster_xyz.pkl",'wb'))
    # xyz=pickle.load(open("cluster_xyz.pkl",'rb')) 

    # pcd=open3d.geometry.PointCloud()
    # pcd.points=open3d.utility.Vector3dVector(xyz)
    # pcd.colors=open3d.utility.Vector3dVector(np.ones_like(xyz))    

    # vis = open3d.visualization.Visualizer()
    # vis.create_window(width=1620, height=1080)    
    # vis.get_render_option().background_color = np.asarray([0, 0, 0])
    # vis.add_geometry(pcd) 
    # vis.run() 

    # control=vis.get_view_control()
    # params=control.convert_to_pinhole_camera_parameters()
    # I,E=params.intrinsic.intrinsic_matrix, params.extrinsic    

    # img=np.asarray(vis.capture_screen_float_buffer())
    # points_i=np.array([ project_point(I,E, point) for point in xyz ] )

    # # rvec,_=cv2.Rodrigues(E[0:3,0:3])
    # # tvec=1*E[0:3,3]
    # # print(E)
    # # print(tvec)
    # # points_i,_=cv2.projectPoints(xyz,rvec,tvec,I,None)
    # # points_i=points_i.reshape((-1,2))   

    # #points_norm=np.linalg.norm(points_i, axis=1) #Didn't work!
    # mask=np.bitwise_and.reduce(( points_i[:,0]>=0, points_i[:,0]<=IMAGE_WIDHT, points_i[:,1]>=0, points_i[:,1]<=IMAGE_HEIGHT  )) #Works! <== Use this to keep it simple
    # #mask=np.bitwise_and.reduce(( points_i[:,0]>=-IMAGE_WIDHT, points_i[:,0]<=2*IMAGE_WIDHT, points_i[:,1]>=-IMAGE_HEIGHT, points_i[:,1]<=2*IMAGE_HEIGHT  )) #Also ok!
    # points_i=points_i[mask,:]

    # for p in points_i[::10]:
    #     _=cv2.circle(img, (int(p[0]), int(p[1])), 6, (0,255,0))   

    # rect=cv2.minAreaRect(points_i[:,0:2].astype(np.float32))
    # box=np.int0(cv2.boxPoints(rect))
    # _=cv2.drawContours(img,[box],0,(255,255,0),thickness=2)         
    # print('rect',rect)
    # cv2.imshow("",img); cv2.waitKey()

    # quit()
    ###New verify as above: raw class points -> clustered
    
    ### BBox verify after clustering
    # options=CLUSTERING_OPTIONS[label]
    # xyz=load_files_for_label('data/numpy/'+scene_name, label=CLASSES_DICT[label], max_points=None)
    # pcd=open3d.geometry.PointCloud()
    # pcd.points=open3d.utility.Vector3dVector(xyz)
    # down_pcd=pcd.voxel_down_sample(voxel_size=0.02)
    # xyz=np.asarray(down_pcd.points)     
    # xyz=semantic.utils.reduce_points(xyz, int(1e5)) 
    # cluster=DBSCAN(eps=options['eps'], min_samples=options['min_samples'], leaf_size=30, n_jobs=-1).fit(xyz)
    # #cluster=DBSCAN(eps=options['eps'], min_samples=500, leaf_size=30, n_jobs=-1).fit(xyz)
    
    # vis=visualize_dbscan_o3d(xyz, cluster.labels_)
    # control=vis.get_view_control()
    # params=control.convert_to_pinhole_camera_parameters()
    # I,E=params.intrinsic.intrinsic_matrix, params.extrinsic

    # #boxes_w=[open3d.geometry.OrientedBoundingBox.create_from_points(open3d.utility.Vector3dVector( xyz[cluster.labels_ == idx] )) for idx in range(np.max(cluster.labels_)+1) ]

    # img=np.asarray(vis.capture_screen_float_buffer())
    # #for idx in range(np.max(cluster.labels_)+1):
    # for idx in (2,):
    #     points_w=xyz[cluster.labels_ == idx]
    #     #points_w=get_hull_points(points_w)
    #     points_i=np.array([ project_point(I,E, point) for point in points_w ])
    #     mask=np.bitwise_and.reduce(( points_i[:,0]>=0, points_i[:,0]<=IMAGE_WIDHT, points_i[:,1]>=0, points_i[:,1]<=IMAGE_HEIGHT  )) #Works! <== Use this to keep it simple
    #     points_i=points_i[mask, :]

    #     # rvec,_=cv2.Rodrigues(E[0:3,0:3])
    #     # tvec=1*E[0:3,3]
        
    #     # points_i,_=cv2.projectPoints(xyz,rvec,tvec,I,None)
    #     # points_i=points_i.reshape((-1,2))
    #     # points_i[:,0]=points_i[:,0]

    #     for p in points_i:
    #          _=cv2.circle(img, (int(p[0]), int(p[1])), 6, (0,255,0))       

    #     #bbox=np.int0( (np.min(points_i[:,0]), np.min(points_i[:,1]), np.max(points_i[:,0]), np.max(points_i[:,1])) )   
    #     #_=cv2.rectangle( img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,255,0))  

    #     rect=cv2.minAreaRect(points_i[:,0:2].astype(np.float32))
    #     box=np.int0(cv2.boxPoints(rect))
    #     cv2.drawContours(img,[box],0,(255,255,0),thickness=2)

    #     cv2.imshow("",img); cv2.waitKey() 
    # quit()    
    # ### BBox verify after clustering   

    
    # hulls=[]
    # for i in range(np.max(cluster.labels_)+1):
    #     gem=open3d.geometry.PointCloud()
    #     gem.points=open3d.utility.Vector3dVector(xyz[cluster.labels_ == i])
    #     hull,_=gem.compute_convex_hull()
    #     hulls.append(hull)

    # for hull in hulls:
    #     points_w=np.asarray(hull.vertices)
    #     points_i=np.array( [project_point(I,E,point) for point in points_w ] )
    #     points_i=np.int0(points_i)
    #     _=cv2.rectangle(img, (np.min(points_i[:,0]), np.min(points_i[:,1])), (np.min(points_i[:,0]), np.min(points_i[:,1])), (0,0,255), thickness=2 )
    #     for p in points_i:
    #         _=cv2.circle(img, (p[0],p[1]), 2, (0,0,255))

    
    # quit()

'''
CONVEX HULL bis projected oder ganz durch ist das LETZTE! Sonst via Images / nachfragen
'''