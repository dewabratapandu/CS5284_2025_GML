import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def pairwise_euclidean_from_scratch(X):
    """
    Computes the pairwise Euclidean distance matrix for a data matrix X from scratch.
    
    Args:
        X: A NumPy array of shape (n_samples, n_features).
        
    Returns:
        A square NumPy array of shape (n_samples, n_samples) with the distances.
    """
    # 1. Calculate the squared L2 norm for each row (sample).
    # This corresponds to the ||a||^2 and ||b||^2 terms.
    X_squared_norms = np.sum(X**2, axis=1)
    
    # 2. Calculate the dot product term (a . b) for all pairs.
    # This is done efficiently with a single matrix multiplication.
    dot_product = np.dot(X, X.T)
    
    # 3. Use the formula ||a-b||^2 = ||a||^2 - 2a.b + ||b||^2.
    # We use broadcasting to expand X_squared_norms to the correct shape.
    sq_distances = X_squared_norms[:, np.newaxis] - 2 * dot_product + X_squared_norms
    
    # 4. Handle potential floating-point inaccuracies that might result in
    #    very small negative numbers.
    sq_distances = np.maximum(sq_distances, 0)
    
    # 5. Take the square root to get the final Euclidean distances.
    distances = np.sqrt(sq_distances)
    
    return distances

def pairwise_cosine_from_scratch(X):
    """
    Computes the pairwise cosine distance matrix for a data matrix X from scratch.
    
    Args:
        X: A NumPy array of shape (n_samples, n_features).
        
    Returns:
        A square NumPy array of shape (n_samples, n_samples) with the distances.
    """
    # 1. Calculate the dot product between all pairs of vectors.
    dot_product = np.dot(X, X.T)
    
    # 2. Calculate the L2 norm (magnitude) of each vector.
    norms = np.linalg.norm(X, axis=1)
    
    # 3. Calculate the outer product of the norms to get the ||a|| * ||b|| terms.
    #    Add a small epsilon for numerical stability to avoid division by zero.
    norm_product = np.outer(norms, norms)
    epsilon = 1e-8
    
    # 4. Calculate the cosine similarity.
    similarity = dot_product / (norm_product + epsilon)
    
    # 5. The cosine distance is 1 - cosine similarity.
    distances = 1 - similarity
    
    return distances

def pairwise_manhattan_from_scratch(X):
    """
    Computes the pairwise Manhattan distance matrix for a data matrix X from scratch.
    
    Args:
        X: A NumPy array of shape (n_samples, n_features).
        
    Returns:
        A square NumPy array of shape (n_samples, n_samples) with the distances.
    """
    # Use broadcasting to compute the absolute difference between all pairs of rows.
    # X[:, np.newaxis, :] has shape (n_samples, 1, n_features)
    # X[np.newaxis, :, :] has shape (1, n_samples, n_features)
    # The result of the subtraction has shape (n_samples, n_samples, n_features)
    abs_diff = np.abs(X[:, np.newaxis, :] - X[np.newaxis, :, :])
    
    # Sum along the last axis (the features) to get the Manhattan distance.
    distances = np.sum(abs_diff, axis=-1)
    
    return distances

def pairwise_kl_divergence_from_scratch(X):
    """
    Computes the pairwise KL Divergence matrix for a data matrix X.
    Each row in X is treated as a probability distribution.
    
    Args:
        X: A NumPy array of shape (n_samples, n_features) where each
           row sums to 1.
        
    Returns:
        A square NumPy array of shape (n_samples, n_samples) where the
        element D[i, j] is the KL divergence from row i to row j.
    """
    # Add a small epsilon for numerical stability to avoid log(0)
    epsilon = 1e-10
    X_safe = X + epsilon
    
    # 1. Compute the first term: sum(P * log(P)) for each row P.
    # This is the negative entropy of each distribution.
    term1 = np.sum(X_safe * np.log(X_safe), axis=1)
    
    # 2. Compute the second term: sum(P * log(Q)) for all pairs.
    # This is efficiently calculated as a matrix multiplication.
    term2 = np.dot(X, np.log(X_safe).T)
    
    # 3. Combine the terms. The result D[i, j] = term1[i] - term2[i, j].
    # Broadcasting term1[:, np.newaxis] makes it compatible for subtraction.
    kl_divergence = term1[:, np.newaxis] - term2
    
    return kl_divergence

def pairwise_wasserstein1d_from_scratch(X):
    """
    Computes the pairwise 1D Wasserstein distance matrix for a data matrix X.
    Each row in X is treated as a 1D probability distribution.
    
    Args:
        X: A NumPy array of shape (n_samples, n_features) where each
           row sums to 1.
        
    Returns:
        A square NumPy array of shape (n_samples, n_samples) with the distances.
    """
    n_samples = X.shape[0]
    
    # 1. Pre-compute the Cumulative Distribution Function (CDF) for each row.
    cdfs = np.cumsum(X, axis=1)
    
    # 2. Initialize the distance matrix.
    distances = np.zeros((n_samples, n_samples))
    
    # 3. Iterate through all pairs to compute the distance.
    for i in range(n_samples):
        for j in range(n_samples):
            # The Wasserstein-1D distance is the sum of the absolute
            # differences between the CDFs.
            distances[i, j] = np.sum(np.abs(cdfs[i] - cdfs[j]))
            
    return distances

# --- Example Usage and Verification ---

# Create some random data
X_test = np.random.rand(5, 3)

# Calculate distances using our from-scratch function
D_scratch = pairwise_euclidean_from_scratch(X_test)

# Calculate distances using scikit-learn for comparison
D_sklearn = pairwise_distances(X_test, metric='euclidean')

print("--- From Scratch Result ---")
print(D_scratch)

print("\n--- Scikit-learn Result ---")
print(D_sklearn)

# Verify that the two results are numerically very close
are_close = np.allclose(D_scratch, D_sklearn)
print(f"\nResults are the same: {are_close}")