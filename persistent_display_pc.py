import numpy as np
from numpy import inf
np.float = np.float64
import os
from matplotlib import pyplot as plt
import trimesh
from scipy.spatial.transform import Rotation as R
import cv2
from trimesh.viewer.windowed import (SceneViewer,
                        render_scene)

import ast
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import argparse
import trimesh

from sensor_msgs.msg import JointState
from std_msgs.msg import String

# camera intrinsic values
K = [901.208740234375, 0.0, 648.2962646484375, 0.0, 900.9511108398438, 379.364013671875, 0.0, 0.0, 1.0]
#K = [901.208740234375, 0.0, 648.2962646484375, 0.0, 900.9511108398438, 379.364013671875, 0.0, 0.0, 1.0]
K = [600.8057861328125, 0.0, 325.5308532714844, 0.0, 600.634033203125, 252.90933227539062, 0.0, 0.0, 1.0]
fx, _, cx, _, fy, cy, _, _, _ = K
K = np.array(K).reshape(3, 3)

# based on our spreadsheets we identify that certain object types can be classified into the object
whitelisted_categories = {
    'banana': ['banana', 'surfboard'], 
    'bowl': ['bowl', 'toilet', 'cup'], 
    'bread_knife': ['knife'], 
    'fork': ['fork'], 
    'green_cup': ['cup', 'toilet'], 
    'green_pan': ['cup'], 
    'hammer': ['toothbrush', 'bird'], 
    'lock': ['cup'], 
    'mustard_bottle': ['sports ball', 'bottle'], 
    'mustard': ['sports ball', 'frisbee', 'banana', 'cup'], 
    'purple_cup': ['cup', 'toilet'], 
    'red_screwdriver': ['baseball bat'], 
    'scissor': ['scissors'], 
    'scissors_occluded': ['scissors'], 
    'scissors_occluded_purple': ['scissors'], 
    'spoon': ['spoon'], 
    'wine_glass': ['wine glass', 'cup'], 
    'f_cup': ["bowl", "frisbee", "disc", "toilet"],
    'mug': ["frisbee", "disc", "toilet", "cup", "potted_plant", "stop sign", 'vase'],
    'mug2': ["frisbee", "disc", "toilet", "cup", "potted_plant", "stop sign", 'vase'],
    'mug3': ["frisbee", "disc", "toilet", "cup", "potted_plant", "stop sign", 'vase'],
    'e_cup': ["frisbee"],
    'plate': ['frisbee', 'apple'],
    'spatula': ['vase', 'knife'],
    'biscuits': ['book', 'kite'],
    'biscuit': ['book', 'kite', 'cup'],
    'pliers': ['scissors']
}
# color for pointcloud labelling
colour_mappings = {
    'wine glass': [0, 0, 0, 30],
    'banana': [255, 0, 0, 30],
    'fork': [0, 255, 0, 30],
    'tv': [0, 0, 255, 30],
    'dining table': [255, 255, 0, 30],
    'bowl': [255, 0, 255, 30],
    'cup': [0, 255, 255, 30],
    'knife': [0, 0, 0, 30],
    'bottle': [0, 0, 0, 30],
    'null': [0, 0, 0, 30],
}

def farthest_points(data,
                    nclusters,
                    dist_func,
                    return_center_indexes=False,
                    return_distances=False,
                    verbose=False):
    """
      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of 
          clusters.
        return_distances: bool, If True, return distances of each point from centers.

      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in 
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of 
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(data.shape[0],
                             dtype=np.int32), np.arange(data.shape[0],
                                                        dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0], ), dtype=np.int32) * -1
    distances = np.ones((data.shape[0], ), dtype=np.float32) * 1e7
    centers = []

    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter

        if verbose:
            print('farthest points max distance : {}'.format(
                np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)
    return clusters


def create_gripper_marker(color=[0, 0, 255], tube_radius=0.001, sections=6, scale = 1.):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=0.002*scale,
        sections=sections,
        segment=[
            [4.10000000e-02*scale, -7.27595772e-12*scale, 6.59999996e-02*scale],
            [4.10000000e-02*scale, -7.27595772e-12*scale, 1.12169998e-01*scale],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002*scale,
        sections=sections,
        segment=[
            [-4.100000e-02*scale, -7.27595772e-12*scale, 6.59999996e-02*scale],
            [-4.100000e-02*scale, -7.27595772e-12*scale, 1.12169998e-01*scale],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002*scale, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02*scale]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002*scale,
        sections=sections,
        segment=[[-4.100000e-02*scale, 0, 6.59999996e-02*scale], [4.100000e-02*scale, 0, 6.59999996e-02*scale]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    return tmp

def q_multiply(q1, q0):
    x0, y0, z0, w0 = q0
    x1, y1, z1, w1 = q1
    return np.array([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                     -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0], dtype=np.float64)

def distance_by_translation_point(p1, p2):
    """
      Gets two nx3 points and computes the distance between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))


def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates whether to use farthest point sampling
      to downsample the points. Farthest point sampling version runs slower.
    """
    
    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(pc, npoints, distance_by_translation_point, return_center_indexes=True)
        else:
            center_indexes = np.random.choice(range(pc.shape[0]), size=npoints, replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc


def backproject(depth_cv, intrinsic_matrix, return_finite_depth=True, return_selection=False):
    '''
    Input:
        - depth_cv (matrix): depth map
        - intrinsic_matrix (matrix): camera configuration
    Return:
        - pointcloid (Nx3)
    '''

    depth = depth_cv.astype(np.float32, copy=True)
    depth[depth == 0] = inf
    depth[depth > 30000] = inf

    # get intrinsic matrix
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width*height, 3)

    # backprojection
    R = np.dot(Kinv, x2d.transpose())

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width*height), (3, 1)), R)
    X = np.array(X).transpose()
    if return_finite_depth:
        selection = np.isfinite(X[:, 0])
        X = X[selection, :]

    if return_selection:
        return X, selection
        
    return X

def q2matrix(Q):

    # Extract the values from Q
    q0 = Q[3]
    q1 = Q[0]
    q2 = Q[1]
    q3 = Q[2]
    
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                        [r10, r11, r12],
                        [r20, r21, r22]])
    
    return rot_matrix

def compute_shifted_grasp(t, q, distance=0.1):
    q0, q1, q2, q3 = q
    x, y, z = t
    
    p  = np.array([x,y,z])

    rot = q2matrix([q0, q1, q2, q3])
    new_p  = np.dot(rot,np.array([[0],[0],[distance]])).squeeze() + p

    return new_p, q

def load_transforms(filename):
    transforms = []
    with open(filename, 'r') as f:
        for l in f:
            translation = list(map(float, l[16:-2].split(', ')))
            rotation = list(map(float, f.readline()[27:-2].split(', ')))
            transform_matrix = np.eye(4)
            transform_matrix[:3,:3] = R.from_quat(rotation).as_matrix()
            transform_matrix[:3,3] = translation
            transforms.append(transform_matrix)
    return transforms

    


THRESHOLD = 50


def load_scene_filter(num_views, masks):
    scene = trimesh.Scene()
    all_pc = []
    RGB = np.load('rgb.npy')
    depth = np.load('depth.npy')
    transforms = load_transforms('ch_transforms.txt')
    global_mask = np.ones((480, 640))
    global_mask[0:80, 0:80] = 0
    global_mask = global_mask.astype(bool)  
    cv2.imwrite('mask.png', global_mask.astype(np.uint8)*255)
    print(num_views)
    for view in range(len(num_views)):
        pointclouds = []
        if num_views[view] != 1:
            continue
        print("rendering..")
        RGB_image = RGB[view] # obtain the RGB image for the blank region
        RGB_image = cv2.cvtColor(RGB_image, cv2.COLOR_BGR2RGB)
        RGBD_image = depth[view] # obtain the RGBD image

        RGB_image[~global_mask,:] = [0,0,0]
        RGB_image[~masks[view][0],:] = [0,0,0]

        RGBD_image[~global_mask] = [0]
        RGBD_image[~masks[view][0]] = [0]
        
        RGBD_depths = RGBD_image.reshape(480*640)
        mask = ~(RGBD_depths == 0)
        RGB_colours = RGB_image.reshape(480*640, 3)
        RGB_colours = RGB_colours[mask]
        
        # pc = create_pc(image_fin)
        pc = backproject(RGBD_image, K)
        pc /= 1000


        cloud = trimesh.points.PointCloud(pc, colors=RGB_colours)
        cloud.apply_transform(transforms[view])
        scene.add_geometry(cloud)
        all_pc.append(cloud.vertices)

    all_pc = np.vstack(all_pc)
    
    # scene = trimesh.Scene()
    # cloud = trimesh.points.PointCloud(all_pc, colors=np.tile(np.array([255 - 0 * 30, 0 * 30, 0, 10]), (len(all_pc), 1)))
    # scene.add_geometry(cloud)
    return scene, all_pc


def callback(callback_period):
    global all_grasps
    taste_the_rainbow = [
        [255, 0, 0],
        [255, 127, 0],
        [255, 255, 0],
        [0, 255, 0],
        [0, 255, 255], 
        [0, 0, 255],
        [127, 0, 255],
        [255, 0, 255],
        [255, 0, 127],
        [0, 0, 0],
    ]
    try:
        # return
        for i in range(len(all_grasps)):
            scene.delete_geometry(f"grasp {i}")
        scene.delete_geometry('item')

        pc = np.load("/home/crslab/cehao/data/pc/pc_segmented.npy")
        colours = np.load("/home/crslab/cehao/data/pc/colours.npy")
        cloud = trimesh.points.PointCloud(pc, colors=colours)
        scene.add_geometry(cloud, geom_name="item")
        all_grasps = []
        all_t = np.load("/home/crslab/cehao/data/grasp/tran.npy")
        all_r = np.load("/home/crslab/cehao/data/grasp/rot.npy")
        selected_index = 3
        for i in range(len(all_t)):
            all_grasps.append(make_gripper(all_t[i], all_r[i], taste_the_rainbow[i] if i < len(taste_the_rainbow) else [255, 255, 255]))

        for i in range(len(all_grasps)):
            scene.add_geometry(all_grasps[i], geom_name=f"grasp {i}")
    except Exception as e:
        print(e)
        pass

def make_gripper(t, r, color):
    q = R.from_quat(r).as_quat()
    s = R.from_euler('xyz', [0,   0 ,  90], degrees=True).as_quat()
    fake_q = q_multiply(q, s)
    t_x = np.eye(4)

    if color == [70, 255, 255]:
        # t = compute_shifted_grasp(t, fake_q, distance=0.13)[0]
        t_x[:3, :3] = R.from_quat(r).as_matrix()
        t_x[:3, 3] = t
    else:
        # t = compute_shifted_grasp(t, fake_q, distance=0.112169998)[0]
        t_x[:3, :3] = R.from_quat(r).as_matrix()
        t_x[:3, 3] = t

    g = create_gripper_marker(color=color, scale=1).apply_transform(t_x)
    
    return g


class PCViewer(SceneViewer):
    def __init__(self, scene, callback=None, callback_period=0.1):
        super(PCViewer, self).__init__(scene)
        self.grasps = []
        

    def on_key_press(self, symbol, modifiers):
        global all_grasps
        if symbol == 97:
            print("refreshing grasps")
            
            all_grasps = grasps
            super(PCViewer, self).on_key_press(symbol, modifiers)
            

all_grasps = []
if __name__ == "__main__":
    taste_the_rainbow = [
        [255, 0, 0],
        [255, 127, 0],
        [255, 255, 0],
        [0, 255, 0],
        [0, 255, 255], 
        [0, 0, 255],
        [127, 0, 255],
        [255, 0, 255],
        [255, 0, 127],
        [0, 0, 0],
    ]

    grasp_index = 0
    THRESHOLD = 0
    
    transforms = load_transforms('ch_transforms.txt')
    
    num_views = 5
    # if item[:-2][-1] == '5':
    #     pc_pa = item[:-2][:-2]
    # else:
    #     pc_pa = item[:-3]
        
    scene = trimesh.Scene()
    pc = np.load("/home/crslab/cehao/data/pc/pc_segmented.npy")
    colours = np.load("/home/crslab/cehao/data/pc/colours.npy")
    cloud = trimesh.points.PointCloud(pc, colors=colours)
    scene.add_geometry(cloud, geom_name="item")
    all_grasps = []
    all_t = np.load("/home/crslab/cehao/data/grasp/tran.npy")
    all_r = np.load("/home/crslab/cehao/data/grasp/rot.npy")
    for i in range(len(all_t)):
        all_grasps.append(make_gripper(all_t[i], all_r[i], [70, 255, 255] if i != 0 else [255, 0, 0]))
    print(len(all_grasps))
    for i in range(len(all_grasps)):
            scene.add_geometry(all_grasps[i], geom_name=f"grasp {i}")

    #grasps = np.load(f"/home/crslab/kaiqichen_project/data/real_robot/{item}_grasp.npy")
    SceneViewer(scene, callback=callback, callback_period=1)
    