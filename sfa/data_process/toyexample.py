import torch

# Define the grid cell size
grid_cell_size = 50 / 608

# Create the x grid
x_grid = torch.arange(0, 50, grid_cell_size)

# Create the y grid
y_grid = torch.arange(-25, 25, grid_cell_size)

selected_points = torch.tensor([
    [0.0, -25.0, 5.0, 1, 0.1, 0.0, 0.0, 0.0],
    [0.0, -25.0, 4.0, 1, 0.6, 10.0, 10.0, 10.0],
    [0.0, -25.0, 3.0, 1, 0.5, 100.0, 100.0, 100.0],
    [0.0, -25.0, 2.0, 1, 0.4, 255.0, 255.0, 255.0],
    [0.0, -25.0, 1.0, 1, 0.9, 100.0, 100.0, 100.0],
    [0.0822, -25.0, 1.0, 1, 0.9, 100.0, 100.0, 100.0],
    [50.0, 25.0, 4.0, 1, 55.0, 255.0, 255.0, 255.0],
    [50.0, 25.0, 2.0, 1, 52.0, 1.0, 1.0, 1.0],
    [50.0, 25.0, 2.0, 1, 52.0, 1.0, 1.0, 1.0],
    [50.0, 25.0, 3.0, 1, 45.0, 10.0, 10.0, 10.0]
], dtype=torch.float32)

# Calculate the grid indices for x and y
x_grid_indices = torch.argmin(torch.abs(x_grid.view(1, -1) - selected_points[:, 0].view(-1, 1)), dim=1)
y_grid_indices = torch.argmin(torch.abs(y_grid.view(1, -1) - selected_points[:, 1].view(-1, 1)), dim=1)

print(f"The x coordinates are in the grid cells at indices {x_grid_indices}.")
print(f"The y coordinates are in the grid cells at indices {y_grid_indices}.")

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

# Print the maximum and mean values in each column
print("Maximum and Mean values in each column:")
print(max_values)
print(mean_values)

exp_factor = 0.1
# Divide each column by its respective maximum value, handling zero denominators
normalized_points_z = torch.where(max_values != 0, selected_z / max_values, torch.zeros_like(selected_z))
# Calculate normalized points_i
normalized_points_i = torch.where(
    non_zero_mask, 
    torch.exp(-exp_factor * torch.abs(selected_i - mean_values)), 
    torch.zeros_like(selected_i)
)

# Print the normalized points
print("Normalized points_z:")
print(normalized_points_z)
print("Normalized points_i:")
print(normalized_points_i)

# Calculate the mean values in each column
sum_values_z = torch.sum(normalized_points_z, dim=0)
sum_values_i = torch.sum(normalized_points_i, dim=0)
print("sum values z in each column:")
print(sum_values_z)
print("sum values i in each column:")
print(sum_values_i)

weight_factor_z = torch.where(max_values != 0, normalized_points_z / sum_values_z, torch.zeros_like(normalized_points_z))
weight_factor_i = torch.where(max_values != 0, normalized_points_i / sum_values_i, torch.zeros_like(normalized_points_z))
print("Weight factor z:")
print(weight_factor_z)
print("Weight factor i:")
print(weight_factor_i)

# Multiply the two matrices element-wise
result_z = weight_factor_z * selected_r
result_i = weight_factor_i * selected_r

# Print the result
print("Element-wise multiplication result:")
print(result_z)
print(result_i)

sum_result_z = torch.sum(result_z, dim=0)
sum_result_i = torch.sum(result_i, dim=0)
print("sum_result_z:")
print(sum_result_z)
print("sum_result_i:")
print(sum_result_i)

density = 0.8
density_r = density * sum_result_i + (1-density) * sum_result_z

print (density_r)
target_shape = (608, 608)
rMap = torch.zeros((608, 608))
rMap = density_r.view(target_shape)
print (rMap)