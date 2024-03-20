import numpy as np
import pandas as pd

selected_points = [
    [0.0, -25.0, 5.0, 1, 0.1, 0.0, 0.0, 0.0],
    [0.0, -25.0, 4.0, 1, 0.6, 10.0, 10.0, 10.0],
    [0.0, -25.0, 3.0, 1, 0.5, 100.0, 100.0, 100.0],
    [0.0, -25.0, 2.0, 1, 0.4, 255.0, 255.0, 255.0],
    [0.0, -25.0, 1.0, 1, 0.9, 100.0, 100.0, 100.0],
    [0.0825, -25.0, 1.0, 1, 0.9, 100.0, 100.0, 100.0],
    [50.0, 25.0, 4.0, 1, 55.0, 255.0, 255.0, 255.0],
    [50.0, 25.0, 2.0, 1, 52.0, 1.0, 1.0, 1.0],
    [50.0, 25.0, 2.0, 1, 52.0, 1.0, 1.0, 1.0],
    [50.0, 25.0, 3.0, 1, 45.0, 10.0, 10.0, 10.0]
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

# Group by grid cells and count points in each group
grouped_1 = df.groupby(['grid_x', 'grid_y']).size().reset_index(name='point_count')

rMap = np.zeros((608, 608))
gMap = np.zeros((608, 608))
bMap = np.zeros((608, 608))
dMap = np.zeros((608, 608))

# Fill in the density map with point counts
for index, row in grouped_1.iterrows():
    x_idx = row['grid_x']
    y_idx = row['grid_y']
    count = row['point_count']
    dMap[x_idx, y_idx] = 0.8


# Calculate normalized values for 'z' and 'intensity' using vectorized operations
df['normalized_z'] = df['z'] / df.groupby(['grid_x', 'grid_y'])['z'].transform('max')
df['normalized_z'] /= df.groupby(['grid_x', 'grid_y'])['normalized_z'].transform('sum')

df['normalized_intensity'] = df['intensity'] - df.groupby(['grid_x', 'grid_y'])['intensity'].transform('mean')
df['normalized_intensity'] = np.exp(-0.1 * np.abs(df['normalized_intensity']))
df['normalized_intensity'] /= df.groupby(['grid_x', 'grid_y'])['normalized_intensity'].transform('sum')

# Calculate intermediate values using vectorized operations
dMap_multiplied = dMap[df['grid_x'], df['grid_y']] * df['normalized_intensity']
intermediate_values = dMap_multiplied + (1 - dMap[df['grid_x'], df['grid_y']]) * df['normalized_z']

df['r'] = df['r'] * intermediate_values
df['g'] = df['g'] * intermediate_values
df['b'] = df['b'] * intermediate_values
# Calculate grouped means
grouped_means = df.groupby(['grid_x', 'grid_y'])[['r', 'g', 'b']].sum().reset_index()

print(df)
print (grouped_means)

rMap[grouped_means['grid_x'], grouped_means['grid_y']] = grouped_means['r']
gMap[grouped_means['grid_x'], grouped_means['grid_y']] = grouped_means['g']
bMap[grouped_means['grid_x'], grouped_means['grid_y']] = grouped_means['b']

print("\nrMap:")
print(rMap)
print("\ngMap:")
print(gMap)
print("\nbMap:")
print(bMap)
print("\ndMap:")
print(dMap)



