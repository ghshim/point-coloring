import os
import sys
import cv2
import numpy as np
import pandas as pd
import torch
from pytictoc import TicToc

# Check if a GPU is available, and if not, fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    labeled_data = torch.cat((lidar_points, torch.arange(len(lidar_points)).reshape(-1, 1).to(device)), dim=1)
    return labeled_data

def add_v_labels(lidar_points, device):
    # Create a tensor with the same shape as lidar_points[1]
    indices = torch.arange(lidar_points.shape[1], device=device)
    
    # Add the indices as an additional row to lidar_points
    labeled_data = torch.cat((lidar_points, indices.unsqueeze(0)), dim=0)
    
    return labeled_data

def makeBEVMap(PointCloud_,Camera_,Calib_, boundary):
    t.tic()
    Height = cnf.BEV_HEIGHT + 1
    Width = cnf.BEV_WIDTH + 1
    
    Camera_ = torch.tensor(Camera_, dtype=torch.float32, device=device)
    PointCloud_ = torch.tensor(PointCloud_, dtype=torch.float32, device=device)
    png = Camera_.clone()
    calib = Calibration(Calib_)
    P2 = torch.tensor(calib.P2, dtype=torch.float32, device=device)
    R0_rect = torch.tensor(calib.R0, dtype=torch.float32, device=device)
    value_0 = torch.tensor([0, 0, 0], device=device).unsqueeze(0)  
    value_1 = torch.tensor([0, 0, 0, 1], device=device).unsqueeze(1)  
    R0_rect = torch.cat((R0_rect, value_0), dim=0)
    R0_rect = torch.cat((R0_rect, value_1), dim=1)
    Tr_velo_to_cam = torch.tensor(calib.V2C, dtype=torch.float32, device=device)
    Tr_velo_to_cam = torch.cat((Tr_velo_to_cam, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=device)), dim=0)

    scan = PointCloud_[:, :4].clone()
    scan[:, 2] -= 2.73
    ones_column = torch.ones((scan.shape[0], 1), dtype=torch.float32, device=device)
    velo = torch.cat((scan[:, :3], ones_column, scan[:, 3:]), dim=1).t()
    cam = torch.matmul(torch.matmul(torch.matmul(P2, R0_rect), Tr_velo_to_cam), velo[:4, :])
    cam_labeled = add_v_labels(cam, device)
    mask = cam_labeled[2, :] >= 0
    cam_labeled = torch.masked_select(cam_labeled, mask).reshape(cam_labeled.shape[0], -1)

    # get u,v,z
    cam_labeled[:2] /= cam_labeled[2,:]
    IMG_H,IMG_W,_ = png.shape

    # filter point out of canvas
    u, v, _ = cam_labeled[:3, :]
    valid_indices = (
        (u > 0) & (u < IMG_W) & (v > 0) & (v < IMG_H)
    )
    cam_labeled = cam_labeled[:, valid_indices]

    # Obtain RGB values at projected locations
    u, v, _ = cam_labeled[:3, :] 
    u = u.to(torch.int64)
    v = v.to(torch.int64)
    # Access rgb_values on the GPU
    rgb_values = png[v, u]
    labeled = cam_labeled[-1]  

    # Discretize Feature Map
    PointCloud = PointCloud_.clone()
    PointCloud[:, 0] = (PointCloud[:, 0] / cnf.DISCRETIZATION).floor().to(torch.int32)
    PointCloud[:, 1] = ((PointCloud[:, 1] / cnf.DISCRETIZATION) + Width / 2).floor().to(torch.int32)
    PointCloud[:, 2] += 0.0001
    labels = torch.arange(len(PointCloud), device=device)
    PointCloud = torch.cat((PointCloud, labels.view(-1, 1)), dim=1)
    selected_mask = torch.isin(PointCloud[:, 4], labeled)
    selected = PointCloud[selected_mask]
    selected_points = torch.cat((selected, rgb_values.view(-1, 3)), dim=1)

    # # sort-3times for Height, Intensity & Density
    sorted_indices = torch.argsort((PointCloud[:, 0]))
    print (sorted_indices)
    print (sorted_indices.shape)
    PointCloud = PointCloud[sorted_indices]
    # Extract unique rows based on the first two columns
    unique_elements, _, unique_counts = torch.unique(PointCloud[:, 0:2], dim=0, return_inverse=True, return_counts=True)
    # Find the indices of the unique rows in the original tensor
    unique_indices = torch.tensor([torch.where(torch.all(PointCloud[:, 0:2] == row, dim=1))[0][0] for row in unique_elements])
    PointCloud_top = PointCloud[unique_indices]

    # Height Map, Intensity Map, Density Map & Color Map
    heightMap = torch.zeros((Height, Width), device=device)
    intensityMap = torch.zeros((Height, Width), device=device)
    densityMap = torch.zeros((Height, Width), device=device)

    # some important problem is image coordinate is (y,x), not (x,y)
    max_height = abs(boundary['maxZ'] - boundary['minZ'])
    indices_x = torch.arange(PointCloud_top.shape[0]).to(torch.long)
    indices_y = torch.arange(PointCloud_top.shape[0]).to(torch.long)
    normalized_heights = PointCloud_top[:, 2] / max_height
    heightMap[indices_x, indices_y] = normalized_heights
    
    unique_counts = torch.tensor(unique_counts, dtype=torch.float32)
    normalizedCounts = torch.minimum(torch.tensor(1.0), torch.log(unique_counts + 1.0) / torch.log(torch.tensor(64.0)))
    intensityMap[indices_x, indices_y] = PointCloud_top[:, 3]
    densityMap[indices_x, indices_y] = normalizedCounts

    # Color point grid cells
    grid_cell_size = 50.0 / 608.0
    x_grid = torch.arange(0.0, 50.0, grid_cell_size, device=device)
    y_grid = torch.arange(-25.0, 25.0, grid_cell_size, device=device)

    # Add grid cell information to the DataFrame (you can replace this with PyTorch operations if available)
    x_grid = torch.tensor(x_grid, dtype=torch.float32)
    y_grid = torch.tensor(y_grid, dtype=torch.float32)

    # Calculate the grid indices for x and y
    x_grid_indices = torch.argmin(torch.abs(x_grid.view(1, -1) - selected_points[:, 0].view(-1, 1)), dim=1)
    y_grid_indices = torch.argmin(torch.abs(y_grid.view(1, -1) - selected_points[:, 1].view(-1, 1)), dim=1)

    grid_indices = 608 * x_grid_indices + y_grid_indices

    selected_z = torch.zeros(selected_points.shape[0], 369664)
    selected_i = torch.zeros(selected_points.shape[0], 369664)
    selected_r = torch.zeros(selected_points.shape[0], 369664)
    selected_g = torch.zeros(selected_points.shape[0], 369664)
    selected_b = torch.zeros(selected_points.shape[0], 369664)

    # Assign the values from the third column of selected_points to selected_z
    selected_z[torch.arange(selected_points.shape[0]), grid_indices] = selected_points[:, 2]
    selected_i[torch.arange(selected_points.shape[0]), grid_indices] = selected_points[:, 4]
    selected_r[torch.arange(selected_points.shape[0]), grid_indices] = selected_points[:, 5]
    selected_g[torch.arange(selected_points.shape[0]), grid_indices] = selected_points[:, 6]
    selected_b[torch.arange(selected_points.shape[0]), grid_indices] = selected_points[:, 7]

    # Find the maximum value in each column
    max_values = torch.max(selected_z, dim=0).values

    # Create a mask to identify non-zero values
    non_zero_mask = selected_i != 0.0

    # Calculate the sum of non-zero values along each column and replace with 1 if it's 0
    sum_non_zero = torch.sum(non_zero_mask, dim=0)
    sum_non_zero = torch.where(sum_non_zero == 0, torch.tensor(1), sum_non_zero)

    # Calculate the mean values, ignoring zero values
    mean_values = torch.sum(selected_i * non_zero_mask, dim=0) / sum_non_zero
    
    exp_factor = 0.1

    # Divide each column by its respective maximum value, handling zero denominators
    normalized_points_z = torch.where(max_values != 0, selected_z / max_values, torch.zeros_like(selected_z))
    # Calculate normalized points_i
    normalized_points_i = torch.where(
        non_zero_mask, 
        torch.exp(-exp_factor * torch.abs(selected_i - mean_values)), 
        torch.zeros_like(selected_i)
    )
    # Calculate the mean values in each column
    sum_values_z = torch.sum(normalized_points_z, dim=0)
    sum_values_i = torch.sum(normalized_points_i, dim=0)

    weight_factor_z = torch.where(max_values != 0, normalized_points_z / sum_values_z, torch.zeros_like(normalized_points_z))
    weight_factor_i = torch.where(max_values != 0, normalized_points_i / sum_values_i, torch.zeros_like(normalized_points_z))
    
    # Multiply the two matrices element-wise
    result_z_r = weight_factor_z * selected_r
    result_i_r = weight_factor_i * selected_r
    result_z_g = weight_factor_z * selected_g
    result_i_g = weight_factor_i * selected_g
    result_z_b = weight_factor_z * selected_b
    result_i_b = weight_factor_i * selected_b

    sum_result_z_r = torch.sum(result_z_r, dim=0)
    sum_result_i_r = torch.sum(result_i_r, dim=0)
    sum_result_z_g = torch.sum(result_z_g, dim=0)
    sum_result_i_g = torch.sum(result_i_g, dim=0)
    sum_result_z_b = torch.sum(result_z_b, dim=0)
    sum_result_i_b = torch.sum(result_i_b, dim=0)

    density = normalizedCounts.view(1, -1)

    density_r = density * sum_result_i_r + (1-density) * sum_result_z_r
    density_g = density * sum_result_i_g + (1-density) * sum_result_z_g
    density_b = density * sum_result_i_b + (1-density) * sum_result_z_b
    target_shape = (608, 608)
    rMap = torch.zeros((608, 608))
    gMap = torch.zeros((608, 608))
    bMap = torch.zeros((608, 608))
    rMap = density_r.view(target_shape)
    gMap = density_g.view(target_shape)
    bMap = density_b.view(target_shape)

    RGB_Map = torch.zeros((6, Height - 1, Width - 1), device=device)
    RGB_Map[5, :, :] = rMap
    RGB_Map[4, :, :] = gMap
    RGB_Map[3, :, :] = bMap
    RGB_Map[2, :, :] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # r_map
    RGB_Map[1, :, :] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # g_map
    RGB_Map[0, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map
    t.toc()
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
    print (x, y, w, l)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    cv2.polylines(img, [corners_int], True, color, 2)
    corners_int = bev_corners.reshape(-1, 2).astype(int)
    cv2.line(img, (corners_int[0, 0], corners_int[0, 1]), (corners_int[3, 0], corners_int[3, 1]), (255, 255, 0), 2)