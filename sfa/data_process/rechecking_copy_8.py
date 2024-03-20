import torch

# Example matrix (replace this with your own matrix)
matrix = torch.tensor([[1, 2, 10],
                       [3, 4, 20],
                       [7, 8, 30],
                       [5, 6, 40]])

# Extract the values from the first and second columns
first_column_values = torch.tensor([1, 3, 5, 7])  # Replace with the values from the first column
second_column_values = torch.tensor([2, 4, 6, 8])  # Replace with the values from the second column

# Create an empty result matrix filled with zeros
result_matrix = torch.zeros(len(first_column_values), len(second_column_values), dtype=matrix.dtype)

# Find the corresponding indices where the first and second columns match
indices = torch.where((matrix[:, 0:1] == first_column_values.view(-1, 1)) &
                      (matrix[:, 1:2] == second_column_values.view(-1, 1)))

# Fill the result matrix with the matching values from the third column at diagonal positions
result_matrix[indices[0], indices[0]] = matrix[indices[0], 2]

# Print the resulting matrix
print(result_matrix)

