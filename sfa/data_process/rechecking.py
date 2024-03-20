import numpy as np

# Example selected_points list with (x, y, z, label, intensity, r, g, b) values
selected_points = [
    (12.5, -8.3, 1.0, 'PointA', 100, 11, 11, 11),
    (12.51, -8.29, 11.0, 'PointA', 50, 255, 0, 0),
    (27.1, -4.7, 2.5, 'PointB', 75, 0, 255, 0),
    (27.11, -4.7, 2.2, 'PointB', 25, 111, 255, 11),
    (3.2, -15.9, 0.8, 'PointC', 50, 0, 0, 255),
    (14.8, -20.1, 3.2, 'PointD', 125, 255, 255, 0),
    (37.6, 9.8, 1.1, 'PointE', 150, 0, 255, 255),
    (50, 25, 1.1, 'PointE', 150, 255, 255, 255),
    (50, 25-50/608, 1.1, 'PointE', 150, 255, 255, 255),
    (0, -25, 1.1, 'PointE', 150, 255, 255, 255),
    (12.51, -8.29, 12.0, 'PointA', 50, 255, 0, 0)
]

# Color point grid cells
grid_cell_size = 50/609
x_grid = np.arange(0, 50 , grid_cell_size)
y_grid = np.arange(-25, 25 , grid_cell_size)

# Create a dictionary to store points for each grid cell
grid_cells_z = {}
grid_cells_i = {}
grid_cells = {}

# Classify points into grid cells
for i in range(len(selected_points)):
    x, y, z, label, intensity, r, g, b = selected_points[i]
    grid_x = np.digitize(x, x_grid) - 1  # Find the corresponding x-grid cell index
    grid_y = np.digitize(y, y_grid) - 1  # Find the corresponding y-grid cell index
    grid_cell = (grid_x, grid_y)

    if grid_cell not in grid_cells_i:
        grid_cells_i[grid_cell] = []

    grid_cells_i[grid_cell].append((x, y, z, label, intensity, r, g, b))

    if grid_cell not in grid_cells_z:
        grid_cells_z[grid_cell] = []

    grid_cells_z[grid_cell].append((x, y, z, label, intensity, r, g, b))

    if grid_cell not in grid_cells:
        grid_cells[grid_cell] = []

    grid_cells[grid_cell].append((x, y, z, label, intensity, r, g, b))

# Find the highest z value and mean intensity in each grid cell
highest_z_values = {}
mean_intensity_values = {}
for grid_cell, points in grid_cells_z.items():
    highest_z = float('-inf')  # Initialize the highest z value to negative infinity
    for point in points:
        x, y, z, label, intensity, r, g, b = point
        highest_z = max(highest_z, z)
        sum_intensity = sum(point[4] for point in points)
        mean_intensity = sum_intensity / len(points)
    mean_intensity_values[grid_cell] = mean_intensity
    highest_z_values[grid_cell] = highest_z

# Multiply z, r, g, b based on z / highest_z for each grid cell
for grid_cell, points in grid_cells_z.items():
    highest_z = highest_z_values[grid_cell]
    scaled_points_z = []
    for i in range(len(points)):
        x, y, z, label, intensity, r, g, b = points[i]
        scaling_factor = np.abs(z  / highest_z) if highest_z != 0.0 else 1.0
        scaled_r_z = r * scaling_factor
        scaled_g_z = g * scaling_factor
        scaled_b_z = b * scaling_factor
        scaled_points_z.append((x, y, z, label, intensity, scaled_r_z, scaled_g_z, scaled_b_z))

    # Replace the original points with the scaled points
    grid_cells_z[grid_cell] = scaled_points_z 

# Calculate the average r, g, and b values for the grid cell
for grid_cell, points in grid_cells_z.items():
    num_points = len(points)
    sum_r_z = sum(point[5] for point in points)
    sum_g_z = sum(point[6] for point in points)
    sum_b_z = sum(point[7] for point in points)
    avg_r_z = sum_r_z / num_points
    avg_g_z = sum_g_z / num_points
    avg_b_z = sum_b_z / num_points

    # Update the mean intensity value in the dictionary for the grid cell
    grid_cells_z[grid_cell] = (avg_r_z, avg_g_z, avg_b_z)

# Print the average intensity in each grid cell
for grid_cell, (avg_r_z, avg_g_z, avg_b_z) in grid_cells_z.items():
    print(f"Grid cell {grid_cell}:  Average Color: ({avg_r_z}, {avg_g_z}, {avg_b_z})")

# Multiply r, g, b based on |intensity-mean_intensity) / mean_intensity| for each grid cell
for grid_cell, points in grid_cells_i.items():
    mean_intensity = mean_intensity_values[grid_cell]
    scaled_points_i = []
    for i in range(len(points)):
        x, y, z, label, intensity, r, g, b = points[i]
        scaling_factor = 1 - np.abs((intensity-mean_intensity) / mean_intensity) if mean_intensity != 0.0 else 1.0
        scaled_r_i = r * scaling_factor
        scaled_g_i = g * scaling_factor
        scaled_b_i = b * scaling_factor
        scaled_points_i.append((x, y, z, label, intensity, scaled_r_i, scaled_g_i, scaled_b_i))

    # Replace the original points with the scaled points
    grid_cells_i[grid_cell] = scaled_points_i 


# Calculate the average r, g, and b values for the grid cell
for grid_cell, points in grid_cells_i.items():
    num_points = len(points)
    sum_r_i = sum(point[5] for point in points)
    sum_g_i = sum(point[6] for point in points)
    sum_b_i = sum(point[7] for point in points)
    avg_r_i = sum_r_i / num_points
    avg_g_i = sum_g_i / num_points
    avg_b_i = sum_b_i / num_points

    # Update the mean intensity value in the dictionary for the grid cell
    grid_cells_i[grid_cell] = (avg_r_i, avg_g_i, avg_b_i)

# Print the average intensity in each grid cell
for grid_cell, (avg_r_i, avg_g_i, avg_b_i) in grid_cells_i.items():
    print(f"Grid cell {grid_cell}:  Average Color: ({avg_r_i}, {avg_g_i}, {avg_b_i})")

for grid_cell in grid_cells_i.keys() & grid_cells_z.keys():
    # Get the average r, g, and b values from grid_cells_i
    avg_r_i, avg_g_i, avg_b_i = grid_cells_i[grid_cell]

    # Get the average r, g, and b values from grid_cells_z
    avg_r_z, avg_g_z, avg_b_z = grid_cells_z[grid_cell]


    # Calculate the average between avg_r_i and avg_r_z
    avg_r = (avg_r_i + avg_r_z) / 2
    avg_g = (avg_g_i + avg_g_z) / 2
    avg_b = (avg_b_i + avg_b_z) / 2

    # Update the mean intensity value in the dictionary for the grid cell
    grid_cells[grid_cell] = (avg_r, avg_g, avg_b)

# Print the updated average color in each grid cell
for grid_cell, (avg_r, avg_g, avg_b) in grid_cells.items():
    print(f"Grid cell {grid_cell}:  Average Color: ({avg_r}, {avg_g}, {avg_b})")



