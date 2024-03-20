import numpy as np

def lidar_to_grid(lidar_points, grid_shape=(609, 609), x_range=(0, 50), y_range=(-25, 25)):
    x_min, x_max = x_range
    y_min, y_max = y_range
    grid_x, grid_y = grid_shape
    x_bins = np.linspace(x_min, x_max, grid_x + 1)
    y_bins = np.linspace(y_min, y_max, grid_y + 1)

    # Quantize the x and y coordinates to grid cells
    x_indices = np.digitize(lidar_points[:, 0], x_bins) - 1
    y_indices = np.digitize(lidar_points[:, 1], y_bins) - 1

    # Sort the lidar points into each grid cell
    grid = [[] for _ in range(grid_x)]
    for x, y, z in lidar_points:
        x_idx = np.clip(np.digitize(x, x_bins) - 1, 0, grid_x - 1)
        y_idx = np.clip(np.digitize(y, y_bins) - 1, 0, grid_y - 1)
        grid[x_idx].append([x, y, z])

    return grid

# Example usage:
# Assuming you have lidar_points as a numpy array containing (x, y, z) lidar data
lidar_points = np.array([[10.0, 5.0, 2.5], [20.0, -15.0, 1.8], [30.0, 0.0, 4.2], ...])
grid = lidar_to_grid(lidar_points)

# Now 'grid' is a list of lists, where each element is a list of lidar points in that grid cell.
# You can access points in a specific cell using grid[x_idx], where x_idx is the cell's x-index.
# For example, to get the lidar points in the cell with x-coordinate 2:
points_in_cell_2 = grid[2,3]
print (points_in_cell_2)
