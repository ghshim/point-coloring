import torch

selected_z = torch.tensor([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
], dtype=torch.float32)

selected_i = torch.tensor([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
], dtype=torch.float32)

selected_r = torch.tensor([
    [255.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 125.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 225.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 225.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 125.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [125.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0]
], dtype=torch.float32)

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