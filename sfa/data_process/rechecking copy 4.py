import numpy as np
import pandas as pd

selected_points = [
    (0, -25, 3.1, 'PointE', 0.1, 0, 0, 0),
    (0, -25, 2.2, 'PointE', 0.6, 10, 10, 10),
    (0, -25, 2.2, 'PointE', 0.45, 100, 100, 100),
    (0, -25, 2.1, 'PointE', 0.4, 255, 255, 255),
    (0, -25, 1.1, 'PointE', 0.9, 100, 100, 100),
    (50, 25, 4, 'PointB', 75, 255, 255, 255),
    (50, 25, 2, 'PointB', 50, 1, 1, 1),
    (50, 25, 3, 'PointB', 25, 10, 10, 10)
]

# Create a DataFrame from the selected points
df = pd.DataFrame(selected_points, columns=['x', 'y', 'z', 'label', 'intensity', 'r', 'g', 'b'])

grid_cell_size = 50 / 608
x_grid = np.arange(0, 50, grid_cell_size)
y_grid = np.arange(-25, 25, grid_cell_size)

# Add grid cell information to the DataFrame
df['grid_x'] = np.digitize(df['x'], x_grid) - 1
df['grid_y'] = np.digitize(df['y'], y_grid) - 1

# Group by grid cells
grouped = df.groupby(['grid_x', 'grid_y'])

# Calculate average RGB values without using a separate function
result_df = grouped.apply(lambda group: group.loc[
    [group['z'].idxmax(), 
     group['intensity'].sub(group['intensity'].mean()).abs().idxmin()], 
    ['r', 'g', 'b']
].mean()).reset_index()

print(result_df)

rMap = np.zeros((608, 608))
gMap = np.zeros((608, 608))
bMap = np.zeros((608, 608))

rMap[result_df['grid_x'], result_df['grid_y']] = result_df['r']
gMap[result_df['grid_x'], result_df['grid_y']] = result_df['g']
bMap[result_df['grid_x'], result_df['grid_y']] = result_df['b']

print("\nrMap:")
print(rMap)
print("\ngMap:")
print(gMap)
print("\nbMap:")
print(bMap)