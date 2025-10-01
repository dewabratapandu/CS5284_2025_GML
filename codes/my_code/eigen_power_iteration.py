import numpy as np

def power_iteration(matrix, num_iterations: int):
    """
    Calculates the dominant eigenvalue and eigenvector of a matrix
    using the Power Iteration method.

    Args:
        matrix: A square NumPy matrix.
        num_iterations: The number of iterations to perform.

    Returns:
        A tuple containing the dominant eigenvalue and eigenvector.
    """
    # Start with a random initial vector
    # Using a vector of ones is a common starting point
    vector = np.ones(matrix.shape[1])
    
    for _ in range(num_iterations):
        # Multiply the matrix by the vector
        matrix_vector_product = np.dot(matrix, vector)
        
        # Calculate the magnitude (L2 norm) of the resulting vector
        vector_magnitude = np.linalg.norm(matrix_vector_product)
        
        # Normalize the vector to prevent its values from growing too large
        vector = matrix_vector_product / vector_magnitude
        
    # Once the eigenvector has converged, calculate the eigenvalue
    # using the Rayleigh quotient: (v^T * A * v) / (v^T * v)
    numerator = np.dot(vector.T, np.dot(matrix, vector))
    denominator = np.dot(vector.T, vector)
    
    eigenvalue = numerator / denominator
    
    return eigenvalue, vector

# --- Example Usage ---

# 1. Create a sample square matrix
A = np.array([
    [4, 1, 1],
    [1, 3, -1],
    [1, -1, 2]
])

# 2. Calculate the dominant eigenvalue and eigenvector from scratch
dominant_eigenvalue, dominant_eigenvector = power_iteration(A, 100)

print("--- From Scratch (Power Iteration) ---")
print(f"Dominant Eigenvalue: {dominant_eigenvalue:.4f}")
print(f"Corresponding Eigenvector: {dominant_eigenvector}")

# 3. For verification, compare with NumPy's built-in function
eigenvalues_np, eigenvectors_np = np.linalg.eig(A)
dominant_index = np.argmax(np.abs(eigenvalues_np))

print("\n--- Using NumPy for Verification ---")
print(f"All Eigenvalues (from NumPy): {eigenvalues_np}")
print(f"Dominant Eigenvalue (from NumPy): {eigenvalues_np[dominant_index]:.4f}")
print(f"Corresponding Eigenvector (from NumPy): {eigenvectors_np[:, dominant_index]}")