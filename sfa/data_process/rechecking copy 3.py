import numpy as np
import pandas as pd

selected_points = [
    (0, -25, 0, 'PointE', 150, 255, 255, 255),
    (0, -25, 1.1, 'PointE', 150, 255, 255, 255),
    (0, -25, 1.1, 'PointE', 140, 235, 235, 235),
    (0, -25, 1.1, 'PointE', 130, 255, 255, 255),
    (12.5, -8.3, 1.0, 'PointA', 100, 11, 11, 11),
    (12.5, -8.3, 4.0, 'PointA', 100, 12, 14, 15),
    (12.51, -8.29, 1, 'PointA', 50, 255, 0, 0),
    (27.1, -4.7, 1.5, 'PointB', 75, 0, 255, 0),
    (50, 25, 4, 'PointB', 75, 255, 255, 255),
]

# Create a DataFrame from the selected points
df = pd.DataFrame(selected_points, columns=['x', 'y', 'z', 'label', 'intensity', 'r', 'g', 'b'])

grid_cell_size = 50 / 608
heigh_level = 1
x_grid = np.arange(0, 50, grid_cell_size)
y_grid = np.arange(-25, 25, grid_cell_size)
z_grid = np.arange(0, 4, heigh_level)

# Add grid cell information to the DataFrame
df['grid_x'] = np.digitize(df['x'], x_grid) - 1
df['grid_y'] = np.digitize(df['y'], y_grid) - 1
df['grid_z'] = np.digitize(df['z'], z_grid) - 1


# Calculate the mean intensity for each grid cell
df['mean_intensity'] = df.groupby(['grid_x', 'grid_y', 'grid_z'])['intensity'].transform('mean')

# Calculate the distance of each point's intensity from the mean intensity of its grid cell
df['distance_to_mean'] = np.abs(df['intensity'] - df['mean_intensity'])

# Find the index of the point with the smallest distance to mean intensity in each grid cell
min_distance_indices = df.groupby(['grid_x', 'grid_y', 'grid_z'])['distance_to_mean'].idxmin()

# Update the representation point for each grid cell
result_df = df.loc[min_distance_indices]

# Set up MultiIndex for result_df
result_df.set_index(['grid_x', 'grid_y', 'grid_z'], inplace=True)

# Print the result
print(result_df)

rMap = np.zeros((608, 608, 4))
bMap = np.zeros((608, 608, 4))
gMap = np.zeros((608, 608, 4))

# Index the arrays using the DataFrame's index values
rMap[result_df.index.get_level_values('grid_x'), result_df.index.get_level_values('grid_y'), result_df.index.get_level_values('grid_z')] = result_df['r'].values
gMap[result_df.index.get_level_values('grid_x'), result_df.index.get_level_values('grid_y'), result_df.index.get_level_values('grid_z')] = result_df['g'].values
bMap[result_df.index.get_level_values('grid_x'), result_df.index.get_level_values('grid_y'), result_df.index.get_level_values('grid_z')] = result_df['b'].values


# for z_layer in range(4):
#     print(f"z_layer = {z_layer}")
#     print(rMap[:, :, z_layer])

# Print the resulting maps
print("\ngMap:")
print(rMap[:, :, 0])
print("\ngMap:")
print(gMap[:, :, 0])
print("\nbMap:")
print(bMap[:, :, 0])
