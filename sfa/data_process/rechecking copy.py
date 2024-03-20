import numpy as np
from collections import defaultdict

selected_points = [
    (0, -25, 3.1, 'PointE', 0.1, 0, 0, 0),
    (0, -25, 2.1, 'PointE', 0.5, 255, 255, 255),
    (0, -25, 1.1, 'PointE', 0.9, 100, 100, 100)
]

grid_cell_size = 50 / 608
x_grid = np.arange(0, 50, grid_cell_size)
y_grid = np.arange(-25, 25, grid_cell_size)

grid_cells_z = defaultdict(list)

for x, y, z, _, intensity, r, g, b in selected_points:
    grid_x = np.digitize(x, x_grid) - 1
    grid_y = np.digitize(y, y_grid) - 1
    grid_cell = (grid_y, grid_x)
    grid_cells_z[grid_cell].append((x, y, z, intensity, r, g, b))

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

    grid_cells_z[grid_cell] = (avg_r_z, avg_g_z, avg_b_z)

for grid_cell, (avg_r_z, avg_g_z, avg_b_z) in grid_cells_z.items():
    print(f"Grid cell {grid_cell}:  Average Color: ({avg_r_z}, {avg_g_z}, {avg_b_z})")
