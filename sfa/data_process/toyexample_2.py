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
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
], dtype=torch.float32)

# Find the maximum value in each column
max_values, _ = torch.max(selected_z, dim=0)

# Print the maximum values for each column
print("Maximum values in each column:")
print(max_values)

# Divide each column by its respective maximum value, handling zero denominators
normalized_points = torch.where(max_values != 0, selected_z / max_values, torch.zeros_like(selected_z))

# Print the normalized points
print("Normalized points:")
print(normalized_points)

# Calculate the mean values in each column
sum_values = torch.sum(normalized_points, dim=0)
print("sum values in each column:")
print(sum_values)

weight_factor = torch.where(max_values != 0, normalized_points / sum_values, torch.zeros_like(normalized_points))
print("Weight factor:")
print(weight_factor)
