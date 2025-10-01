import numpy as np
from scipy.sparse import csr_matrix

def numpy_to_csr_from_scratch(matrix):
    """
    Converts a dense NumPy matrix to the three arrays (data, indices, indptr)
    that define a CSR sparse matrix.
    """
    # Initialize the three lists that will form the CSR components
    data = []
    indices = []
    indptr = [0]  # The index pointer always starts with 0

    # Get the number of rows and columns from the matrix
    num_rows, num_cols = matrix.shape

    # Iterate over each element of the dense matrix
    for i in range(num_rows):
        for j in range(num_cols):
            # If the element is not zero, store its value and column index
            if matrix[i, j] != 0:
                data.append(matrix[i, j])
                indices.append(j)
        
        # At the end of each row, the index pointer is updated to be the
        # current count of non-zero elements found so far.
        indptr.append(len(data))
        
    return np.array(data), np.array(indices), np.array(indptr)

# --- Example Usage ---

# 1. Our original dense matrix
numpy_matrix = np.array([
    [1, 0, 0, 0, 2],  # 2 non-zero elements
    [0, 0, 3, 0, 0],  # 1 non-zero element
    [0, 4, 0, 0, 0],  # 1 non-zero element
    [0, 0, 0, 5, 0]   # 1 non-zero element
])

# 2. Generate the CSR components from scratch
data, indices, indptr = numpy_to_csr_from_scratch(numpy_matrix)

print(f"Original Matrix Shape: {numpy_matrix.shape}\n")
print(f"Data array    (non-zero values): {data}")
print(f"Indices array (column indices): {indices}")
print(f"Indptr array  (row pointers): {indptr}")

# 3. You can now build the SciPy CSR matrix using these components
shape = numpy_matrix.shape
sparse_matrix_from_components = csr_matrix((data, indices, indptr), shape=shape)

print("\nFinal SciPy CSR Matrix representation:")
print(sparse_matrix_from_components)