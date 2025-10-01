import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

def linear_kernel_matrix(X):
    """
    Computes the linear kernel matrix for a 2D data array X.
    K[i, j] = X[i]^T * X[j]
    """
    # This is a single, efficient matrix multiplication
    return np.dot(X, X.T)

def gaussian_kernel_matrix(X, sigma=1.0):
    """
    Computes the Gaussian (RBF) kernel matrix for a 2D data array X.
    
    This uses the vectorized trick for pairwise squared Euclidean distance:
    ||a-b||^2 = ||a||^2 - 2a.b + ||b||^2
    """
    # Calculate pairwise squared Euclidean distances in a vectorized way
    sq_dists = -2 * np.dot(X, X.T)
    sq_norms = np.sum(X**2, axis=1)
    sq_dists += sq_norms[:, np.newaxis] + sq_norms
    
    # Compute the kernel matrix
    gamma = 1 / (2 * (sigma**2))
    return np.exp(-gamma * sq_dists)

def polynomial_kernel_matrix(X, a=1, b=1, c=2):
    """
    Computes the polynomial kernel matrix for a 2D data array X.
    K[i, j] = (a * X[i]^T * X[j] + b)^c
    """
    # The dot product is the linear kernel
    dot_product = np.dot(X, X.T)
    return (a * dot_product + b)**c

# --- Example Usage ---

# Create a sample 2D data array (4 samples, 2 features)
X_data = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8]
])

# Calculate the kernel matrices
linear_K = linear_kernel_matrix(X_data)
gaussian_K = gaussian_kernel_matrix(X_data, sigma=5.0)
poly_K = polynomial_kernel_matrix(X_data, a=1, b=1, c=2)

print(f"Original Data Shape: {X_data.shape}\n")

print(f"--- Linear Kernel Matrix (shape: {linear_K.shape}) ---")
print(linear_K)

print(f"\n--- Gaussian (RBF) Kernel Matrix (shape: {gaussian_K.shape}) ---")
print(gaussian_K)

print(f"\n--- Polynomial Kernel Matrix (shape: {poly_K.shape}) ---")
print(poly_K)

# --- Verification with scikit-learn ---
print("\n--- Verifying RBF Kernel with scikit-learn ---")
# scikit-learn uses gamma, so we calculate it first
gamma_val = 1 / (2 * (5.0**2))
sklearn_K = pairwise_kernels(X_data, metric='rbf', gamma=gamma_val)
print(f"scikit-learn result is the same: {np.allclose(gaussian_K, sklearn_K)}")