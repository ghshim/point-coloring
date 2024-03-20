import numpy as np
import pandas as pd

selected_points = [
    (0, -25, 5, 'PointE', 0.1, 0, 0, 0),
    (0, -25, 4, 'PointE', 0.6, 10, 10, 10),
    (0, -25, 3, 'PointE', 0.5, 100, 100, 100),
    (0, -25, 2, 'PointE', 0.4, 255, 255, 255),
    (0, -25, 1, 'PointE', 0.9, 100, 100, 100),
    (50, 25, 4, 'PointB', 55, 255, 255, 255),
    (50, 25, 2, 'PointB', 50, 1, 1, 1),
    (50, 25, 2, 'PointB', 50, 1, 1, 1),
    (50, 25, 3, 'PointB', 45, 10, 10, 10)
]


df = pd.DataFrame(selected_points, columns=['x', 'y', 'z', 'label', 'intensity', 'r', 'g', 'b'])

grid_cell_size = 50 / 608
x_grid = np.arange(0, 50, grid_cell_size)
y_grid = np.arange(-25, 25, grid_cell_size)

df['grid_x'] = np.digitize(df['x'], x_grid) - 1
df['grid_y'] = np.digitize(df['y'], y_grid) - 1

grouped = df.groupby(['grid_x', 'grid_y'])
grouped_1 = grouped.size().reset_index(name='point_count')

# Density map
dMap = np.minimum(1.0, np.log(grouped_1['point_count'] + 1) / np.log(64))
dMap = dMap.reshape(608, 608)

# Normalized z values
df['normalized_z'] = df['z'] / grouped['z'].transform('max')
df['normalized_z'] /= grouped['normalized_z'].transform('sum')

# Normalized intensity values
exp_factor = 0.1
df['normalized_intensity'] = df.groupby(['grid_x', 'grid_y'])['intensity'].transform(
    lambda x: np.exp(-exp_factor * np.abs(x - x.mean()))
)
df['normalized_intensity'] /= grouped['normalized_intensity'].transform('sum')

# Intermediate values
z_intensity_part = dMap[df['grid_x'], df['grid_y']] * df['normalized_intensity']
z_z_part = (1 - dMap[df['grid_x'], df['grid_y']]) * df['normalized_z']
df['Zr+Ir'] = df['r'] * (z_intensity_part + z_z_part)
df['Zg+Ig'] = df['g'] * (z_intensity_part + z_z_part)
df['Zb+Ib'] = df['b'] * (z_intensity_part + z_z_part)

# Grouped means
grouped_means = df.groupby(['grid_x', 'grid_y'])[['Zr+Ir', 'Zg+Ig', 'Zb+Ib']].mean().reset_index()

# Creating the color maps
rMap = np.zeros((608, 608))
gMap = np.zeros((608, 608))
bMap = np.zeros((608, 608))

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
