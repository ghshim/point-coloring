"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
"""

import os
import sys
import cv2
import numpy as np
from pytictoc import TicToc
from collections import defaultdict

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
    t.tic()
    Height = cnf.BEV_HEIGHT + 1
    Width = cnf.BEV_WIDTH + 1

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
    # print(points.shape)
    velo = np.insert(scan,3,1,axis=1).T
    cam = P2 @ R0_rect @ Tr_velo_to_cam @ velo[:4, :]
    cam_labeled = add_v_labels(cam)

    # get u,v,z
    cam_labeled[:2] /= cam_labeled[2,:]
    IMG_H,IMG_W,_ = png.shape

    # filter point out of canvas
    u, v, _ = cam_labeled[:3, :]
    valid_indices = (
        (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    )
    cam_labeled = cam_labeled[:, valid_indices]

    # Obtain RGB values at projected locations
    u, v, _ = cam_labeled[:3, :] 
    rgb_values = png[v.astype(int), u.astype(int)]
    labeled = cam_labeled[-1]  
    t.toc()
    s = t.tocvalue()
    print (s, 'RGB')

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / cnf.DISCRETIZATION))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / cnf.DISCRETIZATION) + Width / 2)
    labels = np.arange(len(PointCloud))
    PointCloud = np.c_[PointCloud, labels]
    selected = PointCloud[np.isin(PointCloud[:, 4], labeled)]
    selected_points = np.hstack((selected,rgb_values))
    
    # sort-3times for Height, Intensity & Density
    sorted_indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[sorted_indices]
    _, unique_indices, unique_counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[unique_indices]

    # Color point grid cells
    grid_cell_size = 50/608
    x_grid = np.arange(0, 50 , grid_cell_size)
    y_grid = np.arange(-25, 25 , grid_cell_size)

    # Create a dictionary to store points for each grid cell
    grid_cells_z = defaultdict(list)

    # Classify points into grid cells
    for i in range(len(selected_points)):
        x, y, z, _, intensity, r, g, b = selected_points[i]
        grid_x = np.digitize(x, x_grid) - 1 # Find the corresponding x-grid cell index
        grid_y = np.digitize(y, y_grid) - 1 # Find the corresponding y-grid cell index
        grid_cell = (grid_y, grid_x)
        grid_cells_z[grid_cell].append((x, y, z, intensity, r, g, b))

    rMap = np.zeros((Height, Width))
    bMap = np.zeros((Height, Width))
    gMap = np.zeros((Height, Width))
    
    for grid_cell, points in grid_cells_z.items():
        points = np.array(points)
        z_values = points[:, 2].astype(float)
        intensity_values = points[:, 3].astype(float)
        
        color_values = points[:, 4:7].astype(float)  # Convert color values to float
        
        scaling_factors = (
            np.abs(z_values / np.max(z_values)) +
            1 - np.abs((intensity_values - np.mean(intensity_values)) / np.mean(intensity_values))
        ) / 2

        scaled_color_values = color_values * scaling_factors[:, None]

        avg_colors = np.mean(scaled_color_values, axis=0)
        avg_r_z, avg_g_z, avg_b_z = avg_colors
     
        rMap[grid_cell[0], grid_cell[1]] = avg_r_z
        gMap[grid_cell[0], grid_cell[1]] = avg_g_z
        bMap[grid_cell[0], grid_cell[1]] = avg_b_z

    # Height Map, Intensity Map ,Density Map & Color Map
    heightMap = np.zeros((Height, Width))
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))


    # some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(boundary['maxZ'] - boundary['minZ']))
    heightMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 2] / max_height

    normalizedCounts = np.minimum(1.0, np.log(unique_counts + 1) / np.log(64))
    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts


    RGB_Map = np.zeros((6, Height -1 , Width -1 ))
    RGB_Map[5, :, :] = rMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]
    RGB_Map[4, :, :] = gMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]
    RGB_Map[3, :, :] = bMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH] 
    RGB_Map[2, :, :] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # r_map
    RGB_Map[1, :, :] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # g_map
    RGB_Map[0, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map
    # print ('cpu is working ...')
    t.toc()
    z = t.tocvalue()
    print (z, 'Maping')
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
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    cv2.polylines(img, [corners_int], True, color, 2)
    corners_int = bev_corners.reshape(-1, 2).astype(int)
    cv2.line(img, (corners_int[0, 0], corners_int[0, 1]), (corners_int[3, 0], corners_int[3, 1]), (255, 255, 0), 2)

