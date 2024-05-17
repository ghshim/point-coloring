import os
import sys
import cv2
import numpy as np
import pandas as pd
from pytictoc import TicToc
t = TicToc()

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

import config.kitti_config as cnf
from data_process.kitti_data_utils import get_filtered_lidar, Calibration

def get_calib(self, idx):
    calib_file = os.path.join(self.calib_dir, '{:06d}.txt'.format(idx))
    # assert os.path.isfile(calib_file)
    return Calibration(calib_file)

def add_h_labels(lidar_points):
    labeled_data = np.hstack((lidar_points, np.arange(len(lidar_points)).reshape(-1, 1)))
    return labeled_data

def add_v_labels(lidar_points):
    labeled_data = np.vstack((lidar_points, np.arange(len(lidar_points[1]))))
    return labeled_data

def makeBEVMap(PointCloud_,Camera_,Calib_, boundary):
    # t.tic()
    # BEV_HEIGHT = 607 , BEV_WIDTH = 607 
    Height = cnf.BEV_HEIGHT + 1
    Width = cnf.BEV_WIDTH + 1
    
    # This is Transfrom velodyne data (lidar data) to Camera coordinate
    png = np.copy(Camera_)
    calib = Calibration(Calib_)
    P2 = calib.P2
    R0_rect = calib.R0
    R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
    R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)
    Tr_velo_to_cam = calib.V2C
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)
    scan = PointCloud_[:, :4].copy()
    scan[:, 2] -= 2.73
    velo = np.insert(scan,3,1,axis=1).T
    cam = P2 @ R0_rect @ Tr_velo_to_cam @ velo[:4, :]
    cam_labeled = add_v_labels(cam)
    cam_labeled = np.delete(cam_labeled,np.where(cam_labeled[2,:]<=0),axis=1)

    # get u,v,z Camera coordinate
    cam_labeled[:2] /= cam_labeled[2,:]
    IMG_H,IMG_W,_ = png.shape

    # filter lidar point out of canvas
    u, v, _ = cam_labeled[:3, :]
    valid_indices = (
        (u > 0) & (u < IMG_W) & (v > 0) & (v < IMG_H)
    )
    cam_labeled = cam_labeled[:, valid_indices]

    # Obtain RGB values at projected locations
    u, v, _ = cam_labeled[:3, :] 
    rgb_values = png[v.astype(int), u.astype(int)]
    labeled = cam_labeled[-1]  

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / cnf.DISCRETIZATION))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / cnf.DISCRETIZATION) + Width / 2)
    PointCloud[:, 2] += 0.0001 # To prevent zero
    labels = np.arange(len(PointCloud))
    PointCloud = np.c_[PointCloud, labels]
    selected = PointCloud[np.isin(PointCloud[:, 4], labeled)]
    # Add point cloud data with rgb_values
    selected_points = np.hstack((selected,rgb_values))
    
    # sort-3times for Height, Intensity and  Density
    sorted_indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[sorted_indices]
    _, unique_indices, unique_counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[unique_indices]

    # Height Map, Intensity Map ,Density Map based on lidar data
    heightMap = np.zeros((Height, Width))
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    # some important problem is image coordinate is (y,x), not (x,y)
    # This is the method of making heightMap (lidar Only)
    max_height = float(np.abs(boundary['maxZ'] - boundary['minZ']))
    heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height
    
    # This is the method of making densityMap (lidar Only)
    normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    # This is the method of making intensityMap (lidar Only)
    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]

    # Create a DataFrame from the selected points
    df = pd.DataFrame(selected_points, columns=['x', 'y', 'z', 'label', 'intensity', 'r', 'g', 'b'])

    # Color point grid cells
    x_grid = np.arange(0, 607 , 1)
    y_grid = np.arange(0, 607 , 1)
    
    # Add grid cell information to the DataFrame
    df['grid_x'] = np.digitize(df['x'], x_grid)
    df['grid_y'] = np.digitize(df['y'], y_grid)

    # Perform grouped operations to minimize the number of groupby calls
    grouped = df.groupby(['grid_x', 'grid_y'])

    # Calculate normalized height within each grid cell
    max_z_per_group = grouped['z'].transform('max')
    df['normalized_z'] = df['z'] / max_z_per_group
    df['normalized_z'] /= grouped['normalized_z'].transform('sum').replace(0, 0.0001)
    # Hyperparameter for weighting the value based on proximity to the mean intensity
    exp_factor = 0.1

    # Calculate normalized intensity within each grid cell
    mean_intensity_per_group = grouped['intensity'].transform('mean')
    df['normalized_intensity'] = np.exp(-exp_factor * np.abs(df['intensity'] - mean_intensity_per_group))
    df['normalized_intensity'] /= grouped['normalized_intensity'].transform('sum').replace(0, 0.0001)

    # Precompute density factors for optimization
    density_factor = densityMap[df['grid_x'], df['grid_y']]
    normalized_intensity_factor = (1 - density_factor) * df['normalized_intensity']
    normalized_z_factor = density_factor * df['normalized_z']

    # Calculate the intermediate RGB values using precomputed factors
    df['Zr+Ir'] = df['r'] * (normalized_intensity_factor + normalized_z_factor)
    df['Zg+Ig'] = df['g'] * (normalized_intensity_factor + normalized_z_factor)
    df['Zb+Ib'] = df['b'] * (normalized_intensity_factor + normalized_z_factor)

    # Calculate grouped means of grid cell R,G,B seperately
    grouped_means = df.groupby(['grid_x', 'grid_y'])[['Zr+Ir', 'Zg+Ig', 'Zb+Ib']].mean().reset_index()

    # Making the grid cell size for the (lidar + RGB) based rgb_map 
    rMap = np.zeros((Height - 1, Width - 1))
    bMap = np.zeros((Height - 1, Width - 1))
    gMap = np.zeros((Height - 1, Width - 1))
    
    # Put the calculated r, g, b value which calculated by the on the right coordinate of rgb_map
    rMap[grouped_means['grid_x'], grouped_means['grid_y']] = grouped_means['Zr+Ir']
    gMap[grouped_means['grid_x'], grouped_means['grid_y']] = grouped_means['Zg+Ig']
    bMap[grouped_means['grid_x'], grouped_means['grid_y']] = grouped_means['Zb+Ib']
    
    # Making the grid 6 channels RGB_Map to give for YOLO architecture
    RGB_Map = np.zeros((6, Height -1 , Width -1 ))
    RGB_Map[5, :, :] = rMap # (lidar + RGB) based r_map
    RGB_Map[4, :, :] = gMap # (lidar + RGB) based b_map
    RGB_Map[3, :, :] = bMap  # (lidar + RGB) based g_map
    RGB_Map[2, :, :] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # lidar r_map
    RGB_Map[1, :, :] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # lidar g_map
    RGB_Map[0, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # lidar b_map
    
    # t.toc()
    return RGB_Map


# bev image coordinates format
def get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    # front left
    bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

    # rear left
    bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw

    # rear right
    bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

    # front right
    bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

    return bev_corners


def drawRotatedBox(img, x, y, w, l, yaw, color):
    bev_corners = get_corners(x, y, w, l, yaw)
    print (x, y, w, l, yaw, color) # bounding box parameters value
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    cv2.polylines(img, [corners_int], True, color, 2)
    corners_int = bev_corners.reshape(-1, 2).astype(int)
    cv2.line(img, (corners_int[0, 0], corners_int[0, 1]), (corners_int[3, 0], corners_int[3, 1]), (255, 255, 0), 2)