# DISHA BHAVSAR 25CS10049
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import cv2 as cv
import scipy.optimize

# IMPORTING THE HELPER.PY FILE FUNCTIONS --------------------------------------------------------------------------------------------------------------------
def _epipoles(E):
    U, S, V = np.linalg.svd(E)
    e1 = V[-1, :]
    U, S, V = np.linalg.svd(E.T)
    e2 = V[-1, :]

    return e1, e2


def displayEpipolarF(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        x, y = plt.ginput(1, mouse_stop=2)[0]

        xc, yc = int(x), int(y)
        v = np.array([[xc], [yc], [1]])

        l = F @ v
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            raise ValueError('Zero line vector in displayEpipolar')

        l = l / s
        if l[1] != 0:
            xs = 0
            xe = sx - 1
            ys = -(l[0] * xs + l[2]) / l[1]
            ye = -(l[0] * xe + l[2]) / l[1]
        else:
            ys = 0
            ye = sy - 1
            xs = -(l[1] * ys + l[2]) / l[0]
            xe = -(l[1] * ye + l[2]) / l[0]

        ax1.plot(x, y, '*', markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)
        plt.draw()


def _singularize(F):
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U.dot(np.diag(S).dot(V))

    return F


def _objective_F(f, pts1, pts2):
    F = _singularize(f.reshape([3, 3]))
    num_points = pts1.shape[0]
    hpts1 = np.concatenate([pts1, np.ones([num_points, 1])], axis=1)
    hpts2 = np.concatenate([pts2, np.ones([num_points, 1])], axis=1)
    Fp1 = F.dot(hpts1.T)
    FTp2 = F.T.dot(hpts2.T)

    r = 0
    for fp1, fp2, hp2 in zip(Fp1.T, FTp2.T, hpts2):
        r += (hp2.dot(fp1))**2 * (1/(fp1[0]**2 + fp1[1]**2) + 1/(fp2[0]**2 + fp2[1]**2))

    return r


def refineF(F, pts1, pts2):
    f = scipy.optimize.fmin_powell(
        lambda x: _objective_F(x, pts1, pts2), F.reshape([-1]),
        maxiter=100000,
        maxfun=10000
    )

    return _singularize(f.reshape([3, 3]))


def camera2(E):
    U,S,V = np.linalg.svd(E)
    m = S[:2].mean()
    E = U.dot(np.array([[m,0,0], [0,m,0], [0,0,0]])).dot(V)
    U,S,V = np.linalg.svd(E)
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

    if np.linalg.det(U.dot(W).dot(V))<0:
        W = -W

    M2s = np.zeros([3,4,4])
    M2s[:,:,0] = np.concatenate([U.dot(W).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,1] = np.concatenate([U.dot(W).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,2] = np.concatenate([U.dot(W.T).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,3] = np.concatenate([U.dot(W.T).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)

    return M2s


def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, sd = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        x, y = plt.ginput(1, mouse_stop=2)[0]

        xc, yc = int(x), int(y)
        v = np.array([[xc], [yc], [1]])

        l = F @ v
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            raise ValueError('Zero line vector in displayEpipolar')

        l = l / s
        if l[0] != 0:
            xs = 0
            xe = sx - 1
            ys = -(l[0] * xs + l[2]) / l[1]
            ye = -(l[0] * xe + l[2]) / l[1]
        else:
            ys = 0
            ye = sy - 1
            xs = -(l[1] * ys + l[2]) / l[0]
            xe = -(l[1] * ye + l[2]) / l[0]

        ax1.plot(x, y, '*', markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        pc = np.array([[xc, yc]])
        p2 = epipolar_correspondences(I1, I2, F, pc)
        ax2.plot(p2[0,0], p2[0,1], 'ro', markersize=8, linewidth=2)
        plt.draw()


def _projtrans(H, p):
    n = p.shape[1]
    p3d = np.vstack((p, np.ones((1,n))))
    h2d = H @ p3d
    p2d = h2d[:2,:] / np.vstack((h2d[2,:], h2d[2,:]))
    return p2d


def _mcbbox(s1, s2, M1, M2):
    c1 = np.array([[0,0,s1[1],s1[1]], [0,s1[0],0,s1[0]]])
    c1p = _projtrans(M1, c1)
    bb1 = [np.floor(np.amin(c1p[0,:])),
           np.floor(np.amin(c1p[1,:])),
           np.ceil(np.amax(c1p[0,:])),
           np.ceil(np.amax(c1p[1,:]))]

    c2 = np.array([[0,0,s2[1],s2[1]], [0,s2[0],0,s2[0]]])
    c2p = _projtrans(M2, c2)
    bb2 = [np.floor(np.amin(c2p[0,:])),
           np.floor(np.amin(c2p[1,:])),
           np.ceil(np.amax(c2p[0,:])),
           np.ceil(np.amax(c2p[1,:]))]

    bb = np.vstack((bb1, bb2))
    bbmin = np.amin(bb, axis=0)
    bbmax = np.amax(bb, axis=0)
    bbp = np.concatenate((bbmin[:2], bbmax[2:]))

    return bbp


def _imwarp(I, H, bb):
    #minx, miny, maxx, maxy = bb
    #dx, dy = np.arange(minx, maxx), np.arange(miny, maxy)
    #x, y = np.meshgrid(dx, dy)

    #s = x.shape
    #x, y = np.ravel(x), np.ravel(y)
    #pp = _projtrans(la.inv(H), np.vstack((x, y)))
    #x, y = pp[0][:,None].reshape(s), pp[1][:,None].reshape(s)

    s = (int(bb[2]-bb[0]), int(bb[3]-bb[1]))
    I = cv.warpPerspective(I, H, s)

    return I


def warpStereo(I1, I2, M1, M2):
    bb = _mcbbox(I1.shape, I2.shape, M1, M2)

    I1p = _imwarp(I1, M1, bb)
    I2p = _imwarp(I2, M2, bb)

    return I1p, I2p, bb


# END OF HELPER.PY FUNCTIONS --------------------------------------------------------------------------------------------------------------------------------



# IMPLEMENTING THE EIGHT POINT ALGORITHM --------------------------------------------------------------------------------------------------------------------

def eight_point(pts1, pts2, M):
    # Estimates the fundamental matrix using the eight-point algorithm.
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
    
    F_refined_norm = refineF(F_rank2, pts1_norm[:, :2], pts2_norm[:, :2])

    # 6. Un-normalize F
    # Using the formula from the prompt: F_unnorm = T^T * F * T
    F_unnorm = T.T @ F_refined_norm @ T

    return F_unnorm



# FIND EPIPOLAR CORRESPONDENCES --------------------------------------------------------------------------------------------------------------------

def epipolar_correspondences(im1, im2, F, pts1):

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



# COMPUTE ESSENTIAL MATRIX --------------------------------------------------------------------------------------------------------------------

def essential_matrix(F, K1, K2):
    
    # Compute E = K2^T * F * K1
    E = K2.T @ F @ K1
    
    return E



# IMPLEMENT TRIANGULATION --------------------------------------------------------------------------------------------------------------------

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




# 1. Load data
data_corresp = np.load('some_corresp.npz')
pts1, pts2 = data_corresp['pts1'], data_corresp['pts2']
img1 = plt.imread('im1.png')
img2 = plt.imread('im2.png')
M = max(img1.shape) # Scaling factor

# 2. Compute Fundamental Matrix
F = eight_point(pts1, pts2, M)
# displayEpipolarF(img1, img2, F)

# 3. Load temple coordinates and find correspondences
temple_data = np.load('temple_coords.npz')
pts1_temple = temple_data['pts1']
pts2_temple = epipolar_correspondences(img1, img2, F, pts1_temple)
# epipolarMatchGUI(img1, img2, F)

# 4. Load intrinsics and compute Essential Matrix
intrinsics = np.load('intrinsics.npz')
K1, K2 = intrinsics['K1'], intrinsics['K2']
E = essential_matrix(F, K1, K2)
# print("essential matrix:\n", E)

# 5. Define Camera Matrices
# P1 is fixed at the origin
P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1)))))
M2_candidates = camera2(E) # Returns 4 candidates for [R | t]

# 6 & 7. Triangulate and find the correct P2
best_P2 = None
best_pts3d = None
max_positive_depth = -1

for i in range(4):
    M2 = M2_candidates[:, :, i]
    P2 = np.dot(K2, M2)
    pts3d = triangulate(P1, pts1_temple, P2, pts2_temple)
    
    # Check depth: Z coordinate must be positive in both camera frames
    # Transform pts3d to camera 2 frame: P_c2 = M2 * P_world
    pts3d_homo = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    pts3d_c2 = np.dot(M2, pts3d_homo.T).T
    
    # Points in front of both cameras
    in_front = np.logical_and(pts3d[:, 2] > 0, pts3d_c2[:, 2] > 0)
    num_positive = np.sum(in_front)
    
    if num_positive > max_positive_depth:
        max_positive_depth = num_positive
        best_P2 = P2
        best_pts3d = pts3d

def calculate_reprojection_error(pts3d, pts2d, P):
    # Convert 3D points to homogeneous coordinates
    pts3d_h = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    
    # Project back to 2D
    projected_h = (P @ pts3d_h.T).T
    
    # De-homogenize the projected 2D points
    projected_2d = projected_h[:, :2] / projected_h[:, 2:]
    
    # Calculate Euclidean distance (L2 norm) between projected and actual points
    errors = np.linalg.norm(projected_2d - pts2d, axis=1)
    
    return np.mean(errors)

reprojection_error = calculate_reprojection_error(best_pts3d, pts2_temple, best_P2)
# print(f"Reprojection Error: {reprojection_error}")

# 8. Plot the 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(best_pts3d[:, 0], best_pts3d[:, 1], best_pts3d[:, 2], c='blue', marker='.')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Temple Reconstruction')
plt.show()
