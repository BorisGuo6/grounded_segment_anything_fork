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

def load_scene(num_views, class_type, specific_view=None, elevated=False):
    scene = trimesh.Scene()
    all_pc = []
    for view in range(num_views):
        pointclouds = []
#         if view == 2 or view == 3:
#             continue
        bag = rosbag.Bag(f'../hc_pc/{class_type}_{view}.bag')
        #bag = rosbag.Bag(f'detectron_true_{view}.bag')
        for topic, msg, t in bag.read_messages(topics=['/camera/aligned_depth_to_color/image_raw']):
            RGBD_image = ros_numpy.image.image_to_numpy(msg) # obtain the RGBD image

        for topic, msg, t in bag.read_messages(topics=['/detectron2_ros/result']):
            masks = msg.masks
            ids = msg.class_ids
            class_names = msg.class_names
            for i in range(len(masks)):
                mask = masks[i]
                id = class_names[i]
                if id not in whitelisted_categories[class_type]:
                    continue
                image = ros_numpy.image.image_to_numpy(mask) # filter out the image mask from the detectron 2 results
                image_fin = cv2.bitwise_and(RGBD_image, RGBD_image, mask=image)
                #image_fin = RGBD_image
                #pc = create_pc(image_fin)
                pc = backproject(image_fin, K)
                pointclouds.append((pc, id))

        bag.close()
        
        for i in range(len(pointclouds)):
            pc, id = pointclouds[i]
            if id not in colour_mappings:
                id = 'null'
            
            if elevated:
                cutoff_height = 300
                x, y, z = np.split(pc, 3, axis=-1)
                mask = (z <= 380)
                pc = pc[mask[..., 0]]
            pc = regularize_pc_point_count(np.array(pc), 500,)
            pc /= 1000
            
            cloud = trimesh.points.PointCloud(pc, colors=np.tile(np.array([255 - view * 30, view * 30, 0, 30]), (len(pc), 1)))
            cloud.apply_transform(transforms[view])
            scene.add_geometry(cloud)
            all_pc.append(cloud.vertices)
            
    for i in all_pc:
        print(i.shape)
    all_pc = np.vstack(all_pc)
    all_pc = regularize_pc_point_count(all_pc, 400, use_farthest_point=True)
    print(all_pc.shape)
    np.save(f"{class_type}_pc.npy", all_pc)
    scene = trimesh.Scene()
    cloud = trimesh.points.PointCloud(all_pc, colors=np.tile(np.array([255 - 0 * 30, 0 * 30, 0, 30]), (len(all_pc), 1)))
    scene.add_geometry(cloud)
    return scene, all_pc

def load_all(num_views, class_type, specific_view=None):
    scene = trimesh.Scene()        
    cutoff_height = 0
    for view in range(num_views):
        pointclouds = []
#         if view == 2 or view == 3:
#             continue
        bag = rosbag.Bag(f'../hc_pc/{class_type}_{view}.bag')
        #bag = rosbag.Bag(f'detectron_true_{view}.bag')
        for topic, msg, t in bag.read_messages(topics=['/camera/aligned_depth_to_color/image_raw']):
            RGBD_image = ros_numpy.image.image_to_numpy(msg) # obtain the RGBD image
            
        pc = backproject(RGBD_image, K)
        pointclouds.append((pc, 0))
        bag.close()
        
        for i in range(len(pointclouds)):
            pc, id = pointclouds[i]
            if id not in colour_mappings:
                id = 'null'
            pc = regularize_pc_point_count(np.array(pc), 5000,)
            pc /= 1000
            
            cloud = trimesh.points.PointCloud(pc, colors=np.tile(np.array([255 - view * 30, view * 30, 0, 30]), (len(pc), 1)))
            cloud.apply_transform(transforms[view])
            scene.add_geometry(cloud)
    
    return scene
    
def peek_items(class_name, view_no=0, view_all=False):
    if view_all:
        for objects in whitelisted_categories.keys():
            for view in range(5):
                bag = rosbag.Bag(f'../hc_pc/{objects}_{view}.bag')
                for topic, msg, t in bag.read_messages(topics=['/detectron2_ros/result']):
                    masks = msg.masks
                    ids = msg.class_ids
                    class_names = msg.class_names
                    print(objects, class_names)
    else:
        bag = rosbag.Bag(f'../hc_pc/{class_name}_{view_no}.bag')
        for topic, msg, t in bag.read_messages(topics=['/detectron2_ros/result']):
            masks = msg.masks
            ids = msg.class_ids
            class_names = msg.class_names
            print(class_name, class_names)


THRESHOLD = 50


def load_scene_filter(num_views, masks):
    scene = trimesh.Scene()
    all_pc = []
    all_colours = []
    RGB = np.load('rgb.npy')
    depth = np.load('depth.npy')
    transforms = load_transforms('ch_transforms.txt')
    global_mask = np.ones((480, 640))
    global_mask[0:80, 0:80] = 0
    global_mask = global_mask.astype(bool)  
    cv2.imwrite('mask.png', global_mask.astype(np.uint8)*255)
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
        all_colours.append(RGB_colours)

    all_pc = np.vstack(all_pc)
    all_colours = np.vstack(all_colours)
    
    # scene = trimesh.Scene()
    # cloud = trimesh.points.PointCloud(all_pc, colors=np.tile(np.array([255 - 0 * 30, 0 * 30, 0, 10]), (len(all_pc), 1)))
    # scene.add_geometry(cloud)
    return scene, all_pc, all_colours


def load_scene_global(num_views, transforms='ch_transforms.txt'):
    scene = trimesh.Scene()
    all_pc = []
    all_colours = []
    RGB = np.load('rgb.npy')
    depth = np.load('depth.npy')
    transforms = load_transforms(transforms)
    global_mask = np.ones((480, 640))
    global_mask[0:80, 0:80] = 0
    global_mask = global_mask.astype(bool)
    for view in range(num_views):
        print("rendering..")
        RGB_image = RGB[view] # obtain the RGB image for the blank region
        RGB_image = cv2.cvtColor(RGB_image, cv2.COLOR_BGR2RGB)
        RGBD_image = depth[view] # obtain the RGBD image

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
        all_colours.append(RGB_colours)

        

    all_pc = np.vstack(all_pc)
    all_colours = np.vstack(all_colours)
    
    # scene = trimesh.Scene()
    # cloud = trimesh.points.PointCloud(all_pc, colors=np.tile(np.array([255 - 0 * 30, 0 * 30, 0, 10]), (len(all_pc), 1)))
    # scene.show()
    return scene, all_pc, all_colours

def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf, min_z=-np.inf, max_z=np.inf):
    
    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
    bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)

    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    return bb_filter

def make_wire_bbox_cylinders(min_corner, max_corner, radius=0.002, sections=12, color=[200,0,0,255]):
    mn = np.asarray(min_corner, dtype=float)
    mx = np.asarray(max_corner, dtype=float)
    verts = np.array([
        [mn[0], mn[1], mn[2]],
        [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]],
        [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]],
        [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]],
        [mn[0], mx[1], mx[2]],
    ])
    edges = np.array([
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7],
    ])
    cylinders = []
    for a,b in edges:
        p0 = verts[a]
        p1 = verts[b]
        # create cylinder along z and then transform to align p0->p1
        length = np.linalg.norm(p1 - p0)
        if length == 0:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
        # compute transform: move to midpoint and align axis
        direction = (p1 - p0) / length
        midpoint = (p0 + p1) / 2.0
        # compute rotation from cylinder's local z-axis to direction
        z = np.array([0,0,1.0])
        axis = np.cross(z, direction)
        if np.linalg.norm(axis) < 1e-8:
            Rmat = np.eye(3)
        else:
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.dot(z, direction))
            Rmat = trimesh.transformations.rotation_matrix(angle, axis)[:3,:3]
        T = np.eye(4)
        T[:3,:3] = Rmat
        T[:3,3] = midpoint
        cyl.apply_transform(T)
        cyl.visual.vertex_colors = color
        cylinders.append(cyl)
    # combine into a single mesh
    if cylinders:
        return trimesh.util.concatenate(cylinders)
    else:
        return None



def get_segmented_pc_mask(global_pc, global_colours, mask):
    RGB_image = cv2.imread('./images/top_down_img.png')
    RGBD_image = cv2.imread('./images/top_down_depth_img.png', cv2.IMREAD_UNCHANGED)

    transform_matrix = np.eye(4)
    # transform_matrix[:3,:3] = R.from_quat([1.000, -0.000, -0.009, 0.022]).as_matrix()
    # transform_matrix[:3,3] = [0.514, 0.032, 0.763]

    # transform_matrix[:3,:3] = R.from_quat([0.999, 0.005, 0.024, 0.049]).as_matrix()
    # transform_matrix[:3,3] = [0.446, -0.036, 0.432]
    transforms = load_transforms('graspgen.txt')
    transform_matrix = transforms[0]

    RGB_image = cv2.cvtColor(RGB_image, cv2.COLOR_BGR2RGB)
    RGB_image[~mask[0],:] = [0,0,0]
    RGBD_image[~mask[0]] = [0]
    
    RGBD_depths = RGBD_image.reshape(480*640)
    mask = ~(RGBD_depths == 0)


    RGB_colours = RGB_image.reshape(480*640, 3)
    RGB_colours = RGB_colours[mask]
    
    # pc = create_pc(image_fin)
    pc = backproject(RGBD_image, K)
    pc /= 1000

    cloud = trimesh.points.PointCloud(pc, colors=RGB_colours)
    cloud.apply_transform(transform_matrix)

    print(RGB_colours.shape, pc.shape)
    scene = trimesh.Scene()
    scene.add_geometry(cloud)

    # cloud3 = trimesh.points.PointCloud(global_pc, colors=global_colours)
    # scene.add_geometry(cloud3)
    return scene, cloud.vertices, RGB_colours

def get_segmented_pc(global_pc, global_colours, mask):
    RGB_image = cv2.imread('./images/top_down_img.png')
    RGBD_image = cv2.imread('./images/top_down_depth_img.png', cv2.IMREAD_UNCHANGED)

    transform_matrix = np.eye(4)
    # transform_matrix[:3,:3] = R.from_quat([1.000, -0.000, -0.009, 0.022]).as_matrix()
    # transform_matrix[:3,3] = [0.514, 0.032, 0.763]

    transform_matrix[:3,:3] = R.from_quat([0.999, 0.005, 0.024, 0.049]).as_matrix()
    transform_matrix[:3,3] = [0.446, -0.036, 0.432]

    RGB_image = cv2.cvtColor(RGB_image, cv2.COLOR_BGR2RGB)
    RGB_image[~mask[0],:] = [0,0,0]
    RGBD_image[~mask[0]] = [0]
    
    RGBD_depths = RGBD_image.reshape(480*640)
    mask = ~(RGBD_depths == 0)


    RGB_colours = RGB_image.reshape(480*640, 3)
    RGB_colours = RGB_colours[mask]

    
    # pc = create_pc(image_fin)
    pc = backproject(RGBD_image, K)
    pc /= 1000

    

    cloud = trimesh.points.PointCloud(pc)
    cloud.apply_transform(transform_matrix)

    points_inside_box = cloud.vertices
    min_x = points_inside_box[:, 0].min() - 0.01
    max_x = points_inside_box[:, 0].max() + 0.01
    min_y = points_inside_box[:, 1].min() - 0.01
    max_y = points_inside_box[:, 1].max() + 0.01
    min_z = -0.07 #max(0.11, points_inside_box[:, 2].min())
    max_z = points_inside_box[:, 2].max() + 0.01

    print(min_x, max_x, min_y, max_y, min_z, max_z, len(points_inside_box))

    scene = trimesh.Scene()
    scene.add_geometry(cloud)
    scene.add_geometry(trimesh.PointCloud(global_pc, colors=global_colours))
    cyl_mesh = make_wire_bbox_cylinders([min_x, min_y, min_z],
                                   [max_x, max_y, max_z],
                                   radius=0.002)
    scene.add_geometry(cyl_mesh)
    # scene.show()

    inside_box = bounding_box(global_pc, min_x, max_x, min_y, max_y, min_z, max_z)
    points_inside_box = global_pc[inside_box]
    global_colours = global_colours[inside_box]

    print(global_colours.shape, points_inside_box.shape)
    
    scene = trimesh.Scene()
    cloud2 = trimesh.points.PointCloud(points_inside_box, colors=global_colours)
    scene.add_geometry(cloud2)
    return scene, points_inside_box, global_colours

def cue_world_point(x, y):
    ## from the pixel coordinates, obtain the world coordinates of the object placement
    RGB_image = cv2.imread('./images/top_down_img.png')
    RGBD_image = cv2.imread('./images/top_down_depth_img.png', cv2.IMREAD_UNCHANGED)
    transform_matrix = np.eye(4)
    transform_matrix[:3,:3] = R.from_quat([0.999, -0.001, -0.030, -0.018]).as_matrix()
    transform_matrix[:3,3] = [-0.527, -0.057, 0.623]

    mask = np.zeros((480, 640))
    mask[x, y] = 1
    mask = mask.astype(bool)

    RGB_image = cv2.cvtColor(RGB_image, cv2.COLOR_BGR2RGB)
    RGB_image[~mask[0],:] = [0,0,0]
    RGBD_image[~mask[0]] = [0]
    
    RGBD_depths = RGBD_image.reshape(480*640)
    mask = ~(RGBD_depths == 0)


    RGB_colours = RGB_image.reshape(480*640, 3)
    RGB_colours = RGB_colours[mask]
    pc = backproject(RGBD_image, K)
    pc /= 1000

    print(len(pc), pc)
    coords = pc[0]
    coords[2] += 0.1

    return coords






def make_gripper(t, r, color):
    q = R.from_quat(r).as_quat()
    s = R.from_euler('xyz', [0,   0 ,  90], degrees=True).as_quat()
    fake_q = q_multiply(q, s)
    t_x = np.eye(4)

    if color == [70, 255, 255]:
        t = compute_shifted_grasp(t, fake_q, distance=0.13)[0]
        t_x[:3, :3] = R.from_quat(r).as_matrix()
        t_x[:3, 3] = t
    else:
        #t = compute_shifted_grasp(t, fake_q, distance=0.112169998)[0]
        t_x[:3, :3] = R.from_quat(r).as_matrix()
        t_x[:3, 3] = t

    g = create_gripper_marker(color=color, scale=1).apply_transform(t_x)
    
    return g

if __name__ == "__main__":
    taste_the_rainbow = [
        [255, 0, 0, 100],
        [255, 127, 0, 100],
        [255, 255, 0, 100],
        [0, 255, 0, 100],
        [0, 255, 255, 100], 
        [0, 0, 255, 100],
        [127, 0, 255, 100],
        [255, 0, 255, 100],
        [255, 0, 127, 100],
        [0, 0, 0, 100],
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
    scene, all_pc  = load_scene_filter(5, "test")

    #grasps = np.load(f"/home/crslab/kaiqichen_project/data/real_robot/{item}_grasp.npy")
    # c
    # scene.show()