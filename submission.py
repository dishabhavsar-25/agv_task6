"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
from scipy.ndimage import uniform_filter
import helper as hlp
from scipy.linalg import rq

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    # replace pass by your implementation

    N = pts1.shape[0]

    # 1. Normalize points
    # Create the transformation matrix T
    T = np.array([[1.0 / M, 0, 0],
                  [0, 1.0 / M, 0],
                  [0, 0, 1]])

    # Convert to homogeneous coordinates by appending a column of 1s
    pts1_h = np.hstack((pts1, np.ones((N, 1))))
    pts2_h = np.hstack((pts2, np.ones((N, 1))))

    # Apply normalization: x_norm = T * x
    pts1_norm = (T @ pts1_h.T).T
    pts2_norm = (T @ pts2_h.T).T

    # Extract the normalized coordinates
    u1, v1 = pts1_norm[:, 0], pts1_norm[:, 1]
    u2, v2 = pts2_norm[:, 0], pts2_norm[:, 1]

    # 2. Construct Matrix A
    # Each row is: [u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, 1]
    A = np.vstack((u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, np.ones(N))).T

    # 3. Solve for F using SVD
    U_a, S_a, Vt_a = np.linalg.svd(A)
    
    # The solution is the last row of V^T (corresponds to the smallest singular value)
    f = Vt_a[-1, :]
    F = f.reshape(3, 3)

    # 4. Enforce Rank-2 Constraint
    U_f, S_f, Vt_f = np.linalg.svd(F)
    
    # Set the smallest singular value to exactly zero
    S_f[-1] = 0
    
    # Reconstruct the rank-2 F matrix
    F_rank2 = U_f @ np.diag(S_f) @ Vt_f
    
    F_refined_norm = hlp.refineF(F_rank2, pts1_norm[:, :2], pts2_norm[:, :2])

    # 6. Un-normalize F
    # Using the formula from the prompt: F_unnorm = T^T * F * T
    F_unnorm = T.T @ F_refined_norm @ T

    return F_unnorm


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    # replace pass by your implementation
    pts2 = np.zeros_like(pts1)
    
    # Define window size (e.g., 15x15 window means a half-width of 7)
    window_size = 15
    w = window_size // 2 
    
    h, width = im1.shape[:2]
    
    # Convert points to homogeneous coordinates
    N = pts1.shape[0]
    pts1_h = np.hstack((pts1, np.ones((N, 1))))
    
    for i in range(N):
        x1, y1 = int(pts1[i, 0]), int(pts1[i, 1])
        
        # 1. Extract the patch from im1
        # Skip if the patch is too close to the image boundary
        if x1 - w < 0 or x1 + w >= width or y1 - w < 0 or y1 + w >= h:
            continue
            
        patch1 = im1[y1-w : y1+w+1, x1-w : x1+w+1]
        
        # 2. Calculate the epipolar line l' = F * x
        l_prime = F @ pts1_h[i].T
        a, b, c = l_prime
        
        best_error = float('inf')
        best_x2, best_y2 = 0, 0
        
        # 3. Search along the epipolar line in im2
        # We can search a reasonable horizontal range. 
        # To optimize, you could restrict this to a neighborhood around x1
        search_range = range(w, width - w) 
        
        for x2 in search_range:
            # Calculate corresponding y2 on the line: ax + by + c = 0
            y2 = int(round(-(a * x2 + c) / b))
            
            # Boundary check for the patch in im2
            if y2 - w < 0 or y2 + w >= h:
                continue
                
            # Extract patch from im2
            patch2 = im2[y2-w : y2+w+1, x2-w : x2+w+1]
            
            # 4. Compute similarity (Euclidean distance / SSD)
            # Make sure to handle potential color channels properly by flattening or summing
            error = np.sum((patch1.astype(float) - patch2.astype(float)) ** 2)
            
            # Update best candidate
            if error < best_error:
                best_error = error
                best_x2 = x2
                best_y2 = y2
                
        # Store the best match
        pts2[i] = [best_x2, best_y2]
        
    return pts2


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    # replace pass by your implementation
    # Compute E = K2^T * F * K1
    E = K2.T @ F @ K1
    
    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    
    N = pts1.shape[0]
    pts3d = np.zeros((N, 3))
    
    for i in range(N):
        u1, v1 = pts1[i, 0], pts1[i, 1]
        u2, v2 = pts2[i, 0], pts2[i, 1]
        
        # Construct the 4x4 A matrix
        A = np.array([
            u1 * P1[2, :] - P1[0, :],
            v1 * P1[2, :] - P1[1, :],
            u2 * P2[2, :] - P2[0, :],
            v2 * P2[2, :] - P2[1, :]
        ])
        
        # Solve for X using SVD
        U, S, Vt = np.linalg.svd(A)
        X_homogeneous = Vt[-1, :]
        
        # De-homogenize the 3D point (divide by the 4th element)
        X = X_homogeneous[:3] / X_homogeneous[3]
        pts3d[i, :] = X
        
    return pts3d


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    # 1. Compute the optical centers
    c1 = -(R1.T @ t1).flatten()
    c2 = -(R2.T @ t2).flatten()
    
    # 2. Compute the new x-axis (parallel to the baseline)
    r1 = (c1 - c2) / np.linalg.norm(c1 - c2)
    
    # 3. Compute the new y-axis (orthogonal to old z and new x)
    # R1[2, :] is the third row of R1, representing the old optical axis (z)
    r2 = np.cross(R1[2, :], r1)
    r2 = r2 / np.linalg.norm(r2)
    
    # 4. Compute the new z-axis (orthogonal to new x and y)
    r3 = np.cross(r1, r2)
    
    # 5. Build the new rotation matrix
    R_new = np.vstack((r1, r2, r3))
    
    R1p = R_new
    R2p = R_new
    
    # We can use K2 as the new intrinsic matrix for both cameras
    K1p = K2.copy()
    K2p = K2.copy()
    
    # Compute the new translation vectors
    t1p = -R_new @ c1.reshape(3, 1)
    t2p = -R_new @ c2.reshape(3, 1)
    
    # 6. Compute the rectification homography matrices M1 and M2
    M1 = K1p @ R_new @ R1.T @ np.linalg.inv(K1)
    M2 = K2p @ R_new @ R2.T @ np.linalg.inv(K2)
    
    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    # replace pass by your implementation
    # Ensure images are float for accurate distance calculation
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)
    
    h, w = im1.shape
    
    # Initialize the disparity map and the minimum cost map
    dispM = np.zeros((h, w), dtype=np.float32)
    min_cost = np.full((h, w), np.inf, dtype=np.float32)
    
    # Loop over all possible disparity values
    for d in range(max_disp + 1):
        # 1. Shift image 2 to the right by 'd' pixels
        # Create a shifted array filled with zeros
        im2_shifted = np.zeros_like(im2)
        if d > 0:
            im2_shifted[:, d:] = im2[:, :-d]
        else:
            im2_shifted = im2.copy()
            
        # 2. Compute the pixel-wise squared difference
        diff = (im1 - im2_shifted) ** 2
        
        # 3. Aggregate the cost over the window size
        # uniform_filter calculates the local mean. We can use it to represent 
        # the sum by implicitly ignoring the constant division factor, 
        # as it doesn't change the argmin location.
        cost = uniform_filter(diff, size=win_size)
        
        # 4. Update the disparity map where the current cost is the lowest
        # Find indices where the new cost is strictly less than the previous minimum
        better_match_mask = cost < min_cost
        
        # Update the minimum cost and the best disparity
        min_cost[better_match_mask] = cost[better_match_mask]
        dispM[better_match_mask] = d
        
    return dispM


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    # 1. Compute the optical centers
    c1 = -(R1.T @ t1).flatten()
    c2 = -(R2.T @ t2).flatten()
    
    # 2. Calculate the baseline (Euclidean distance between camera centers)
    b = np.linalg.norm(c1 - c2)
    
    # 3. Extract the focal length from the intrinsic matrix
    f = K1[0, 0]
    
    # 4. Initialize the depth map
    depthM = np.zeros_like(dispM, dtype=np.float32)
    
    # 5. Create a mask to find where disparity is greater than 0
    # This prevents division by zero.
    valid_pixels = dispM > 0
    
    # 6. Compute depth: Z = (f * b) / d
    depthM[valid_pixels] = (f * b) / dispM[valid_pixels]
    
    return depthM


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
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


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
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
