import numpy as np
import os
import pptk
import time
import sys
import pickle
from .imports import Pose
from main import load_files, view_pptk, resize_window

'''
Module to find the poses and save them
'''

'''
TODO


'''
scene_config={}
scene_config['bildstein_station1_xyz_intensity_rgb']={
    'points':np.array([ [ 40.75127792,  65.6179657 ,   2.50230789],
                        [ 66.08031464,  47.87584305,  -0.11737233],
                        [ 79.23332214,  31.75216103,  -4.95731688],
                        [ 59.31369019,  24.54601097,  -0.2168442 ],
                        [ 53.72531128,  -2.45185494,  -2.87793112],
                        [ 23.58790016, -17.37290764,   1.41321528],
                        [ 16.09674263, -55.24095154,   6.5845623 ],
                        [ 12.94268131, -14.04222393,   2.81412625],
                        [ -7.48832893,   6.72084284,  -1.50921261],
                        [ 17.15963554,  22.26341629,  -0.49522878]]), 
    'num_points': 60
}
scene_config['domfountain_station1_xyz_intensity_rgb']={
    'points':np.array([ [-1.62381992e+01,  4.07225800e+01, -7.98928738e-02],
                        [-1.54097843e+01,  3.69349496e+01, -1.15705913e-01],
                        [-1.45813694e+01,  3.31473191e+01, -1.51518951e-01],
                        [-1.37529545e+01,  2.93596887e+01, -1.87331990e-01],
                        [-1.29245396e+01,  2.55720583e+01, -2.23145029e-01],
                        [-1.20961246e+01,  2.17844279e+01, -2.58958068e-01],
                        [-1.12677097e+01,  1.79967975e+01, -2.94771107e-01],
                        [-1.04392948e+01,  1.42091671e+01, -3.30584146e-01],
                        [-9.61087990e+00,  1.04215367e+01, -3.66397184e-01],
                        [-8.78246498e+00,  6.63390632e+00, -4.02210223e-01],
                        [-7.95405006e+00,  2.84627592e+00, -4.38023262e-01],
                        [-7.12563515e+00, -9.41354485e-01, -4.73836301e-01],
                        [-7.48538601e+00, -3.18791412e+00, -5.16420547e-01],
                        [-8.43921976e+00, -4.66393837e+00, -5.62390396e-01],
                        [-9.39305350e+00, -6.13996262e+00, -6.08360245e-01],
                        [-1.03468873e+01, -7.61598687e+00, -6.54330094e-01],
                        [-1.13007210e+01, -9.09201112e+00, -7.00299943e-01],
                        [-1.22545547e+01, -1.05680354e+01, -7.46269792e-01],
                        [-1.32083885e+01, -1.20440596e+01, -7.92239641e-01],
                        [-1.41622222e+01, -1.35200839e+01, -8.38209491e-01],
                        [-1.51160560e+01, -1.49961081e+01, -8.84179340e-01],
                        [-1.60698897e+01, -1.64721324e+01, -9.30149189e-01],
                        [-1.70237235e+01, -1.79481566e+01, -9.76119038e-01],
                        [-1.76268172e+01, -1.98859749e+01, -9.89305475e-01],
                        [-1.75284308e+01, -2.27473814e+01, -9.36925089e-01],
                        [-1.74300445e+01, -2.56087879e+01, -8.84544702e-01],
                        [-1.73316581e+01, -2.84701944e+01, -8.32164316e-01],
                        [-1.72332717e+01, -3.13316009e+01, -7.79783929e-01],
                        [-1.71348853e+01, -3.41930074e+01, -7.27403543e-01],
                        [-1.70364990e+01, -3.70544139e+01, -6.75023156e-01],
                        [-1.69381126e+01, -3.99158204e+01, -6.22642770e-01],
                        [-1.68397262e+01, -4.27772269e+01, -5.70262383e-01],
                        [-1.67413399e+01, -4.56386334e+01, -5.17881996e-01],
                        [-1.66429535e+01, -4.85000399e+01, -4.65501610e-01],
                        [-1.65445671e+01, -5.13614464e+01, -4.13121223e-01],
                        [-3.31721344e+01, -2.00225105e+01,  6.03923649e-02],
                        [-3.02208022e+01, -1.96297817e+01,  1.36322938e-02],
                        [-2.72694700e+01, -1.92370529e+01, -3.31277773e-02],
                        [-2.43181378e+01, -1.88443241e+01, -7.98878483e-02],
                        [-2.13668056e+01, -1.84515953e+01, -1.26647919e-01],
                        [-1.84154734e+01, -1.80588665e+01, -1.73407990e-01],
                        [-1.54641412e+01, -1.76661377e+01, -2.20168062e-01],
                        [-1.25128091e+01, -1.72734089e+01, -2.66928133e-01],
                        [-9.56147687e+00, -1.68806801e+01, -3.13688204e-01],
                        [-6.61014467e+00, -1.64879513e+01, -3.60448275e-01],
                        [-3.65881248e+00, -1.60952225e+01, -4.07208346e-01],
                        [-7.07480292e-01, -1.57024937e+01, -4.53968417e-01],
                        [ 2.24385190e+00, -1.53097649e+01, -5.00728488e-01],
                        [ 4.62148728e+00, -1.48863351e+01, -4.38896542e-01],
                        [ 6.99912266e+00, -1.44629052e+01, -3.77064596e-01],
                        [ 9.37675804e+00, -1.40394754e+01, -3.15232649e-01],
                        [ 1.17543934e+01, -1.36160456e+01, -2.53400703e-01],
                        [ 1.41320288e+01, -1.31926158e+01, -1.91568757e-01],
                        [ 1.65096642e+01, -1.27691860e+01, -1.29736811e-01],
                        [ 1.88872996e+01, -1.23457562e+01, -6.79048647e-02],
                        [ 2.12649349e+01, -1.19223264e+01, -6.07291857e-03],
                        [ 2.36425703e+01, -1.14988966e+01,  5.57590276e-02],
                        [ 2.60202057e+01, -1.10754668e+01,  1.17590974e-01],
                        [ 2.83978411e+01, -1.06520370e+01,  1.79422920e-01],
                        [ 3.07754765e+01, -1.02286072e+01,  2.41254866e-01]]
    ), 
    'num_points': 60
}
scene_config['neugasse_station1_xyz_intensity_rgb']={
    'points':np.array([[ 1.1074542e+01,  1.0536696e+00,  1.0],
                        [-3.3289959e+00,  1.0078371e-02, 1.0],
                        [-9.9907360e+00, -3.6623685e+00, 1.0],
                        [-3.0083437e+00, -8.5878935e+00, 1.0],
                        [ 1.7781239e+00, -3.6863661e+00, 1.0]]), 
    'num_points': 20
}
scene_config['sg27_station1_intensity_rgb']={
    'points':np.array([ [-108.32433319,  -21.52495956,    4.87378788],
                        [ -40.00649643,  -59.89164352,    2.07816982],
                        [  25.59915161,  -25.89069176,   -1.36464977],
                        [  -8.31031132,   34.38957596,   -0.51066923],
                        [ -75.4786911 ,   40.54391861,    5.46927929]]
    ), 
    'num_points': 20
}
scene_config['sg27_station2_intensity_rgb']={
    'points':np.array([[ -1.19515443e+00, -9.11307907e+00,  9.51060187e-03],
                        [ 1.82419910e+01, -3.71339912e+01, -5.37586331e-01],
                        [ 2.15250740e+01,  1.31393557e+01, -4.16823030e-01],
                        [-1.10820770e+01,  1.10676794e+01,  7.62200296e-01]]
    ), 
    'num_points': 20
}
scene_config['sg27_station4_intensity_rgb']={
    'points':np.array([ [ 43.57066345,  11.67580128,   2.73182034],
                        [  1.99903047,   1.5136199 ,  -0.36759657],
                        [-15.13977814,  -8.66014194,  -0.87346685],
                        [ 13.13288212, -27.31091881,   2.00873756],
                        [ 14.41301632, -55.06941223,   2.5040822 ]]
    ), 
    'num_points': 20
}
scene_config['sg27_station5_intensity_rgb']={
    'points':np.array([ [ 1.34423466e+01,  5.45422173e+01,  3.85324359e-01],
                        [ 2.16214428e+01, -2.28970032e+01, -4.45066392e-01],
                        [-3.56397781e+01, -4.16972542e+01,  2.37969160e-02],
                        [ 4.12746239e+00,  1.53987398e+01,  3.96039844e-01]]
    ), 
    'num_points': 20
}
scene_config['sg27_station9_intensity_rgb']={
    'points':np.array([ [ 2.89740753e+01, -6.17324715e+01,  1.27222669e+00],
                        [ 4.76786661e+00, -6.32071924e+00, -1.17525697e-01],
                        [-4.09983978e+01,  1.98120594e+01, -1.14651370e+00],
                        [ 2.00527477e+01,  8.41629181e+01,  5.48987508e-01],
                        [ 1.16093283e+01,  1.55800877e+01, -5.82260340e-02]]
    ), 
    'num_points': 20
}
scene_config['sg28_station4_intensity_rgb']={
    'points':np.array([ [-27.58380127,  42.79438782,   0.39144072],
                        [-12.6619091 ,   8.91702461,  -0.3901161 ],
                        [ 19.50559235,   1.47677588,  -0.75855958],
                        [ 41.40940475,  -7.858531  ,  -1.27329683],
                        [ 28.13772011, -23.72556877,  -1.0279994 ]]
    ), 
    'num_points': 20
}
scene_config['untermaederbrunnen_station1_xyz_intensity_rgb']={
    'points':np.array([ [-36.01984787,  16.37983131,  -0.98667938],
                        [ -3.9050281 ,  13.9103756 ,  -1.28279018],
                        [  1.42306983,  -0.14167392,  -0.90287954],
                        [ -5.29690695, -10.51343918,  -0.54458594],
                        [ 22.00692368,   8.50140619,  -0.6742084 ]]
    ), 
    'num_points': 40
}


def interpolate_points(points_in, num_points):
    points_in=np.array(points_in)
    assert points_in.shape[1]==3

    if len(points_in)==num_points:
        return points_in

    points=np.zeros((num_points,3))
    for i in range(3):
        points[:,i]=np.interp(np.linspace(0,len(points_in)-1,num_points), np.arange(len(points_in)), points_in[:,i])
    return points

def calculate_poses(viewer, scene_name, points, num_angles, visualize=False):
    time.sleep(1.0)
    poses=[]
    for point in points:
        viewer.set(lookat=point)
        for i_angle,phi in enumerate(np.linspace(np.pi, -np.pi,num_angles+1)[0:-1]): 
            viewer.set(phi=phi,theta=0.0,r=0.0)
            if visualize:
                time.sleep(0.5)

            poses.append( Pose(scene_name, viewer.get('eye'), viewer.get('right'), viewer.get('up'), viewer.get('view'), phi) )

    return poses

def visualize_points(viewer,points,ts=0.2):
    if points.shape[1]==3:
        points=np.hstack(( points,np.array([[np.pi/2,np.pi/2,30],]).repeat(len(points),axis=0) ))
    viewer.play(points,ts=ts*np.arange(len(points)))

if __name__ == "__main__":
    '''
    View pptk -> write config -> interpolate&save poses
    '''    
    scene_name='sg27_station2_intensity_rgb'
    output_path_poses=os.path.join('data','pointcloud_images_o3d_merged',scene_name,'poses.pkl')

    viewer=view_pptk('data/numpy_merged/'+scene_name, remove_artifacts=True, remove_unlabeled=True, max_points=int(15e6))
    resize_window()
    
    num_points=scene_config[scene_name]['num_points']
    points=scene_config[scene_name]['points']
    #quit()
    

    points=interpolate_points(points,num_points)
    # visualize_points(viewer,points)
    # quit()

    #Using viewer-math to set the poses | viewer get's stuck after too many requests / naive work-around
    poses=[]
    viewer.close()
    for start_idx in range(0,len(points),10):
        viewer=view_pptk('data/numpy_merged/'+scene_name, remove_artifacts=True, remove_unlabeled=True, max_points=int(15e6))
        resize_window()
        poses.extend( calculate_poses(viewer, scene_name, points[start_idx:start_idx+10], num_angles=10) )
        viewer.close()

    #poses=calculate_poses(viewer, scene_name, points, num_angles=10)
    print('num poses:',len(poses))
    pickle.dump( poses, open(output_path_poses, 'wb') )
    print('Poses saved!', output_path_poses)


# scene_config['bildstein_station1_xyz_intensity_rgb']={
#     #Nx3 array
#     'points':np.array( [[  6.89986515, -23.8486042 ,   2.24763012],
#                         [ 14.23859787,  -6.84959984,   0.59534997],
#                         [-12.54539108,   8.38919926,  -1.56900454],
#                         [-12.71178436,  25.77069092,  -1.69532895],
#                         [ 12.06625557,  28.25757217,  -1.58685946],
#                         [ 10.43738079,   6.53896713,   0.22520876]]), 
#     'point_size_rgb':0.025, #labels always double
# }
# scene_config['bildstein_station3_xyz_intensity_rgb']={
#     #Nx3 array
#     'points':np.array( [[-13.66415501,  -1.87494004,   1.20972967],
#                         [-15.5337286 ,  18.76299858,   2.06882811],
#                         [  2.10001707,  25.09252548,   2.41423416],
#                         [  7.10345364,  -7.98233128,   0.36108246],
#                         [ 16.39338684, -35.32163239,  -2.28681469]]), 
#     'point_size_rgb':0.025, #labels always double
# }
# scene_config['bildstein_station5_xyz_intensity_rgb']={
#     #Nx3 array
#     'points':np.array( [[-19.88222122,  34.78226089,   1.30118823],
#                         [ 17.61147308,   6.93702793,  -2.1327517 ],
#                         [  7.25930929, -15.78915977,  -1.98583078],
#                         [-12.99044991, -32.33332443,  -1.03720355],
#                         [-31.48099136, -48.11365891,   1.69673967]]), 
#     'point_size_rgb':0.025, #labels always double
# }
# scene_config['domfountain_station1_xyz_intensity_rgb']={
#     #Nx3 array
#     'points':np.array( [[-10.74167538,   10.32442617,  -0.50595516],
#                         [ -4.53943491,   5.41002083,  -0.57776338],
#                         [ -4.10716677,  -3.10575199,   0.08882368],
#                         [-10.06188583,  -7.36892557,   0.50089562],
#                         [-15.61357212,  -5.97002316,   0.73054999],
#                         [-15.33024597,  16.69128513,  -0.38836336]]), 
#     'point_size_rgb':0.025, #labels always double
# }
# scene_config['domfountain_station2_xyz_intensity_rgb']={
#     #Nx3 array
#     'points':np.array( [[-29.57394791,   9.05232334,   0.47919855],
#                         [  4.34464264,  -3.3333323 ,   0.15014482],
#                         [ -0.56595898, -19.08597183,  -0.21233426],
#                         [ -1.47728717, -34.15919495,   0.49576887],
#                         [ 11.51516724, -12.17238903,  -0.28104484],
#                         [ 26.30233955, -10.83860207,  -0.11856288]]), 
#     'point_size_rgb':0.025, #labels always double
# }
# scene_config['domfountain_station3_xyz_intensity_rgb']={
#     #Nx3 array
#     'points':np.array( [[ -7.10327578,  33.63069534,  -0.16973461],
#                         [  0.32602367,   7.58825922,  -0.12918571],
#                         [  4.26744795, -14.16171265,   0.35794488],
#                         [ 23.29409027, -36.65826797,   0.81396568]]), 
#     'point_size_rgb':0.025, #labels always double
# }
# scene_config['sg27_station2_intensity_rgb']={
#     #Nx3 array
#     'points':np.array( [[ -2.28320765, -13.30841255,   0.65572155],
#                         [ 14.10788918, -32.98677063,  -1.28365839],
#                         [ 26.90045547, -23.60473442,  -1.01840901],
#                         [ 23.01133728,  10.0      ,  -1.40091133],
#                         [ -1.34517264,  10.82786083,   0.17833348]]), 
#     'point_size_rgb':0.025, #labels always double
# }

# #TODO: Non-circle ok?
# scene_config['untermaederbrunnen_station1_xyz_intensity_rgb']={
#     #Nx3 array
#     'points':np.array([[-8.15726089e+00, -1.17434473e+01, -2.00530887e-03],
#                         [9.46222305, 3.13939857, 0.07745753],
#                         [-7.56795025,  6.47402334, -0.09737678],
#                         [-19.66889381,  13.40221691,  -0.38015223]]),
#     'point_size_rgb':0.025, #labels always double
# }

# #CARE: z changed to higher
# scene_config['neugasse_station1_xyz_intensity_rgb']={
#     #Nx3 array
#     'points':np.array([[ 1.1074542e+01,  1.0536696e+00,  1.0],
#                         [-3.3289959e+00,  1.0078371e-02, 1.0],
#                         [-9.9907360e+00, -3.6623685e+00, 1.0],
#                         [-3.0083437e+00, -8.5878935e+00, 1.0],
#                         [ 1.7781239e+00, -3.6863661e+00, 1.0]]), 
#     'point_size_rgb':0.025, #labels always double
# }    