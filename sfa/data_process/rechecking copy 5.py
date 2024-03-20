import numpy as np
import pandas as pd

selected_points = [
    (0, -25, 5, 'PointE', 0, 0, 0, 0),
    (0, -25, 5, 'PointE', 0.1, 0, 0, 0),
    (0, -25, 4, 'PointE', 0.6, 10, 10, 10),
    (0, -25, 3, 'PointE', 0.5, 100, 100, 100),
    (0, -25, 2, 'PointE', 0.4, 255, 255, 255),
    (0, -25, 1, 'PointE', 0.9, 100, 100, 100),
    (0, -25, 1, 'PointE', 10, 100, 100, 100),
    (50, 25, 4, 'PointB', 55, 255, 255, 255),
    (50, 25, 2, 'PointB', 52, 1, 1, 1),
    (50, 25, 2, 'PointB', 52, 1, 1, 1),
    (50, 25, 3, 'PointB', 45, 10, 10, 10)
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
    dMap[x_idx, y_idx] = np.minimum(1.0, np.log(count + 1) / np.log(64))


# Calculate normalized values for 'z' based on their sum
df['normalized_z'] = df['z'] / df.groupby(['grid_x', 'grid_y'])['z'].transform('max')
sum_normalized_z = df.groupby(['grid_x', 'grid_y'])['normalized_z'].transform('sum')
sum_normalized_z = sum_normalized_z.replace(0, 0.0001)
df['normalized_z'] /= sum_normalized_z

exp_factor = 0.1  # Adjust this value for desired sensitivity to distance from mean

# Calculate the normalized exponential values for 'intensity' within each grid cell
df['normalized_intensity'] = df['intensity'].groupby([df['grid_x'], df['grid_y']]).transform(lambda x: np.exp(-exp_factor * np.abs(x - x.mean())))
df['normalized_intensity'] /= df.groupby(['grid_x', 'grid_y'])['normalized_intensity'].transform('sum')

# Calculate the intermediate values within each group
df['Zr+Ir'] = df['r'] * (dMap[df['grid_x'], df['grid_y']] * df['normalized_intensity'] + (1-dMap[df['grid_x'], df['grid_y']])*df['normalized_z'])
df['Zg+Ig'] = df['g'] * (dMap[df['grid_x'], df['grid_y']] * df['normalized_intensity'] + (1-dMap[df['grid_x'], df['grid_y']])*df['normalized_z'])
df['Zb+Ib'] = df['b'] * (dMap[df['grid_x'], df['grid_y']] * df['normalized_intensity'] + (1-dMap[df['grid_x'], df['grid_y']])*df['normalized_z'])
# Calculate grouped means
grouped_means = df.groupby(['grid_x', 'grid_y'])[['Zr+Ir', 'Zg+Ig', 'Zb+Ib']].mean().reset_index()

print(df)
print (grouped_means)

rMap[grouped_means['grid_x'], grouped_means['grid_y']] = grouped_means['Zr+Ir']
gMap[grouped_means['grid_x'], grouped_means['grid_y']] = grouped_means['Zg+Ig']
bMap[grouped_means['grid_x'], grouped_means['grid_y']] = grouped_means['Zb+Ib']

print("\nrMap:")
print(rMap)
print("\ngMap:")
print(gMap)
print("\nbMap:")
print(bMap)
print("\ndMap:")
print(dMap)



