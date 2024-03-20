import numpy as np
import pandas as pd

selected_points = [
    (12.5, -8.3, 1.0, 'PointA', 100, 11, 11, 11),
    (12.51, -8.29, 0, 'PointA', 50, 255, 0, 0),
    (27.1, -4.7, 1.5, 'PointB', 75, 0, 255, 0),
    (27.11, -4.7, 2.3, 'PointB', 25, 111, 255, 11),
    (3.2, -15.9, 0.8, 'PointC', 50, 0, 0, 255),
    (14.8, -20.1, 3.2, 'PointD', 125, 255, 255, 0),
    (37.6, 9.8, 1.1, 'PointE', 150, 0, 255, 255),
    (50, 25, 1.1, 'PointE', 150, 255, 255, 255),
    (50, 25-50/608, 1.1, 'PointE', 150, 255, 255, 255),
    (0, -25, 1.1, 'PointE', 150, 255, 255, 255),
    (12.51, -8.29, 3.5, 'PointA', 50, 255, 0, 0),
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

# Calculate scaling factors using Pandas operations
scaling_factors = (
    np.abs(df['z'] / df.groupby(['grid_x', 'grid_y'])['z'].transform('max')) +
    1 - np.abs((df['intensity'] - df.groupby(['grid_x', 'grid_y'])['intensity'].transform('mean')) / df.groupby(['grid_x', 'grid_y'])['intensity'].transform('mean'))
) / 2

# Apply scaling factors to color values
color_columns = ['r', 'g', 'b']
df[color_columns] = df[color_columns].mul(scaling_factors, axis=0)

# Group by grid cells and calculate average scaled color values
result_df = df.groupby(['grid_x', 'grid_y'])[color_columns].mean()

print(result_df)

rMap = np.zeros((608, 608))
bMap = np.zeros((608, 608))
gMap = np.zeros((608, 608))

# Index the arrays using the DataFrame's index values
rMap[result_df.index.get_level_values('grid_x'), result_df.index.get_level_values('grid_y')] = result_df['r'].values
gMap[result_df.index.get_level_values('grid_x'), result_df.index.get_level_values('grid_y')] = result_df['g'].values
bMap[result_df.index.get_level_values('grid_x'), result_df.index.get_level_values('grid_y')] = result_df['b'].values


# Print the resulting maps
print("rMap:")
print(rMap[607,607])
print("\ngMap:")
print(gMap)
print("\nbMap:")
print(bMap)

# # Print the resulting maps
# print("rMap shape:", rMap.shape)
# print("gMap shape:", gMap.shape)
# print("bMap shape:", bMap.shape)