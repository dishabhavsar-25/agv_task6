import numpy as np
from scipy.linalg import rq
import matplotlib.pyplot as plt


def estimate_pose(x, X):
    N = x.shape[0]
    
    # Initialize the A matrix (2N x 12)
    A = np.zeros((2 * N, 12))
    
    for i in range(N):
        u, v = x[i, 0], x[i, 1]
        X_i, Y_i, Z_i = X[i, 0], X[i, 1], X[i, 2]
        
        # Row 1 for point i
        A[2 * i, :] = [X_i, Y_i, Z_i, 1, 
                       0, 0, 0, 0, 
                       -u * X_i, -u * Y_i, -u * Z_i, -u]
        
        # Row 2 for point i
        A[2 * i + 1, :] = [0, 0, 0, 0, 
                           X_i, Y_i, Z_i, 1, 
                           -v * X_i, -v * Y_i, -v * Z_i, -v]
                           
    # Solve A * p = 0 using SVD
    U, S, Vt = np.linalg.svd(A)
    
    # The solution is the last row of V^T (the right singular vector corresponding to the smallest singular value)
    p = Vt[-1, :]
    
    # Reshape the 12x1 vector back into a 3x4 matrix
    P = p.reshape((3, 4))
    
    return P


def estimate_params(P):
    # 1. Split P into M (3x3) and p4 (3x1)
    M = P[:, :3]
    p4 = P[:, 3]
    
    # 2. Perform RQ decomposition on M to get K and R
    K, R = rq(M)
    
    # 3. Enforce positive diagonal elements for the intrinsic matrix K
    # Create a diagonal matrix of the signs of K's diagonal
    T = np.diag(np.sign(np.diag(K)))
    
    # Adjust K and R so the diagonal of K becomes positive
    K = K @ T
    R = T @ R
    
    # Ensure R is a proper rotation matrix (det(R) == +1)
    # If det(R) is -1, it includes a reflection. We flip the signs of both to correct it.
    if np.linalg.det(R) < 0:
        R = -R
        K = -K
        
    # 4. Compute the translation vector t
    # Since p4 = K * t  =>  t = K_inv * p4
    t = np.linalg.inv(K) @ p4
    
    # Reshape t to be a 3x1 column vector to match the requested output format
    t = t.reshape((3, 1))
    
    return K, R, t


def ransac_pose(x, X, num_iters=1000, threshold=2.0):
    """
    Robustly estimates camera matrix P using RANSAC to filter out dynamic objects.
    Assumes `estimate_pose(x, X)` from the previous step is available.
    """
    N = x.shape[0]
    best_P = None
    best_inlier_count = 0
    best_inliers = None
    
    # We need at least 6 points for the DLT camera matrix estimation
    sample_size = 6 
    
    # Convert 3D points to homogeneous coordinates once for fast projection
    X_h = np.hstack((X, np.ones((N, 1))))
    
    for i in range(num_iters):
        # 1. Randomly sample 6 points
        indices = np.random.choice(N, sample_size, replace=False)
        x_sample = x[indices]
        X_sample = X[indices]
        
        # 2. Fit the model hypothesis using the sample
        P_candidate = estimate_pose(x_sample, X_sample)
        
        # 3. Evaluate the model on ALL points
        # Project 3D points to 2D: x_proj = P * X
        x_proj_h = (P_candidate @ X_h.T).T
        
        # De-homogenize (avoiding division by absolute zero)
        z_c = x_proj_h[:, 2:3]
        z_c[z_c == 0] = 1e-6 
        x_proj = x_proj_h[:, :2] / z_c
        
        # 4. Calculate reprojection error (Euclidean distance)
        errors = np.linalg.norm(x - x_proj, axis=1)
        
        # 5. Find inliers
        inliers = np.where(errors < threshold)[0]
        inlier_count = len(inliers)
        
        # 6. Update best model
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_P = P_candidate
            best_inliers = inliers
            
    # 7. Final Polish: Re-estimate P using ALL inliers from the best model
    if best_inliers is not None and len(best_inliers) >= sample_size:
        final_P = estimate_pose(x[best_inliers], X[best_inliers])
    else:
        # Fallback if RANSAC fails entirely
        final_P = best_P 
        
    return final_P, best_inliers



# Load the data from the .npz file
data = np.load('pnp.npz')

# print("Keys in pnp.npz:", data.files)

X_3d = data['X']  # Assuming 'X' holds the 3D points
x_2d = data['x']  # Assuming 'x' holds the 2D image points

# Compute the Camera Pose
P = estimate_pose(x_2d, X_3d)
K, R, t = estimate_params(P)
final_P, inliers = ransac_pose(x_2d, X_3d)



# Output the results
print("\nEstimated camera matrix (P):")
print(np.round(P, 6))
print("\nEstimated Intrinsic Matrix (K):")
print(np.round(K, 6))
print("Estimated Rotation Matrix (R):")
print(np.round(R, 6))
print("\nEstimated Translation Vector (t):")
print(np.round(t, 6))
print("\nRANSAC Estimated camera matrix (final_P):")
print(np.round(final_P, 6))
print("\nNumber of inliers found by RANSAC:", len(inliers))