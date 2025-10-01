import numpy as np

# Create a sample NumPy matrix with many zero elements
numpy_matrix = np.array([
    [1, 0, 0, 0, 2],
    [0, 0, 3, 0, 0],
    [0, 4, 0, 0, 0],
    [0, 0, 0, 5, 0]
])

# Get the row and column indices of non-zero elements
rows, cols = numpy_matrix.nonzero()

# Print the results
print("Original Matrix:")
print(numpy_matrix)

print("\nRow indices of non-zero elements:")
print(rows)

print("\nColumn indices of non-zero elements:")
print(cols)