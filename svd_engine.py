import numpy as np

def compress_channel(channel_matrix, rank):
    """Compresses a single 2D channel using truncated SVD."""
    U, Sigma, Vt = np.linalg.svd(channel_matrix, full_matrices=False)
    
    # Bound the rank to ensure it doesn't exceed matrix dimensions
    max_rank = min(channel_matrix.shape)
    r = min(max(1, rank), max_rank)
    
    # Truncate and reconstruct
    U_trunc = U[:, :r]
    Sigma_trunc = np.diag(Sigma[:r])
    Vt_trunc = Vt[:r, :]
    
    return np.dot(U_trunc, np.dot(Sigma_trunc, Vt_trunc))

def get_energy_data(channel_matrix):
    """Calculates the cumulative energy array using squared singular values."""
    _, Sigma, _ = np.linalg.svd(channel_matrix, full_matrices=False)
    cumulative_energy = np.cumsum(Sigma**2) / np.sum(Sigma**2)
    return cumulative_energy