# import cv2 as cv
# import numpy as np

# # Import your custom functions from your submission file
# # Ensure this file is in the same directory and named correctly
# from submission import (
#     eight_point, essential_matrix, rectify_pair, 
#     get_disparity, get_depth, estimate_pose, estimate_params
# )

# # 3.1 Feature Detection and Tracking
# def detect_features(image):
#     """
#     Detects trackable features using the Shi-Tomasi corner detector.
#     """
#     # Convert to grayscale if the image is in color
#     if len(image.shape) == 3:
#         gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     else:
#         gray = image

#     # Parameters for Shi-Tomasi corner detection
#     # You can tweak these parameters based on your specific image resolution
#     feature_params = dict(maxCorners=300,
#                           qualityLevel=0.01,
#                           minDistance=10,
#                           blockSize=7)

#     # Detect features
#     p0 = cv.goodFeaturesToTrack(gray, mask=None, **feature_params)
    
#     return p0


# # 3.2 PnP
# def estimate_pose(x, X):
#     N = x.shape[0]
    
#     # Initialize the A matrix (2N x 12)
#     A = np.zeros((2 * N, 12))
    
#     for i in range(N):
#         u, v = x[i, 0], x[i, 1]
#         X_i, Y_i, Z_i = X[i, 0], X[i, 1], X[i, 2]
        
#         # Row 1 for point i
#         A[2 * i, :] = [X_i, Y_i, Z_i, 1, 
#                        0, 0, 0, 0, 
#                        -u * X_i, -u * Y_i, -u * Z_i, -u]
        
#         # Row 2 for point i
#         A[2 * i + 1, :] = [0, 0, 0, 0, 
#                            X_i, Y_i, Z_i, 1, 
#                            -v * X_i, -v * Y_i, -v * Z_i, -v]
                           
#     # Solve A * p = 0 using SVD
#     U, S, Vt = np.linalg.svd(A)
    
#     # The solution is the last row of V^T (the right singular vector corresponding to the smallest singular value)
#     p = Vt[-1, :]
    
#     # Reshape the 12x1 vector back into a 3x4 matrix
#     P = p.reshape((3, 4))
    
#     return P

#     # decompose P, K into R, t
# def decompose_projection_matrix(P, K):
#     # 1. Extract M (first 3 columns) and p4 (last column)
#     M = P[:, :3]
#     p4 = P[:, 3]
    
#     # 2. Compute the inverse of K
#     K_inv = np.linalg.inv(K)
    
#     # 3. Extract R and t
#     R_raw = K_inv @ M
#     t = K_inv @ p4
    
#     # 4. Orthogonalize R using SVD (Ensures it is a valid rotation matrix)
#     U, S, Vt = np.linalg.svd(R_raw)
#     R = U @ Vt
    
#     # Ensure a right-handed coordinate system (det(R) should be +1)
#     if np.linalg.det(R) < 0:
#         R = -R
#         t = -t

#     return R, t


# def ransac_eight_point_custom(pts1, pts2, M, num_iters=500, threshold=0.01):
#     """Simple RANSAC wrapper around your custom 8-point algorithm."""
#     N = pts1.shape[0]
#     best_inliers = []
    
#     pts1_h = np.hstack((pts1, np.ones((N, 1))))
#     pts2_h = np.hstack((pts2, np.ones((N, 1))))
    
#     for _ in range(num_iters):
#         # Sample 8 points
#         indices = np.random.choice(N, 8, replace=False)
#         F_cand = eight_point(pts1[indices], pts2[indices], M)
        
#         # Calculate Sampson distance/Epipolar error
#         l2 = (F_cand @ pts1_h.T).T
#         l1 = (F_cand.T @ pts2_h.T).T
        
#         numerator = np.abs(np.sum(pts2_h * l2, axis=1))
#         norm = np.linalg.norm(l2[:, :2], axis=1) + np.linalg.norm(l1[:, :2], axis=1)
#         norm[norm == 0] = 1e-6 # Avoid division by zero
        
#         errors = numerator / norm
#         inliers = np.where(errors < threshold)[0]
        
#         if len(inliers) > len(best_inliers):
#             best_inliers = inliers
            
#     return best_inliers

# def main():
#     # ==========================================
#     # 1. INITIALIZATION & CALIBRATION
#     # ==========================================
#     # Standardized Camera Intrinsic Parameters (K)
#     fx, fy = 517.3, 516.5
#     cx, cy = 318.6, 255.3
#     K = np.array([
#         [fx,  0, cx],
#         [ 0, fy, cy],
#         [ 0,  0,  1]
#     ], dtype=np.float64)

#     # Video setup
#     video_path = 'dataset_video.mp4' # Replace with your actual video path
#     cap = cv.VideoCapture(video_path)
    
#     if not cap.isOpened():
#         print(f"Error: Could not open video file {video_path}")
#         return

#     ret, prev_frame = cap.read()
#     if not ret: return
    
#     prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
    
#     # Normalization parameter M = max(H, W) for 8-point algorithm
#     h_img, w_img = prev_gray.shape
#     M_norm = max(h_img, w_img)
    
#     # Detect initial features
#     p0 = detect_features(prev_frame)

#     pnp = np.load('pnp.npz')

#     X_3d = pnp['X']  # Assuming 'X' holds the 3D points
#     x_2d = pnp['x']  # Assuming 'x' holds the 2D image points

#     # Compute the Camera Pose
#     P = estimate_pose(x_2d, X_3d)
#     R, t = decompose_projection_matrix(P, K)    
#     # Variables for tracking the global trajectory
#     trajectory = []
#     global_R = np.eye(3)
#     global_t = np.zeros((3, 1))
    
#     frame_count = 1
#     max_frames = 50 # Limit to 50 frames for speed; adjust as needed

#     print(f"Processing up to {max_frames} frames...")

#     # ==========================================
#     # 2. MAIN ODOMETRY LOOP
#     # ==========================================
#     while cap.isOpened() and frame_count < max_frames:
#         ret, curr_frame = cap.read()
#         if not ret: break
        
#         curr_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)

#         # --- Step A: Optical Flow Tracking ---
#         p1, st, err = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
        
#         if p1 is not None and len(p1[st == 1]) >= 8:
#             good_new = p1[st == 1]
#             good_old = p0[st == 1]
            
#             # --- Step B: Outlier Rejection ---
#             inliers = ransac_eight_point_custom(good_old, good_new, M_norm)
            
#             if len(inliers) >= 8:
#                 pts1_inliers = good_old[inliers]
#                 pts2_inliers = good_new[inliers]
                
#                 # --- Step C: Epipolar Geometry ---
#                 F = eight_point(pts1_inliers, pts2_inliers, M_norm)
#                 E = essential_matrix(F, K, K)
                
#                 # We use OpenCV here to handle the cheirality check (finding the correct R, t 
#                 # out of the 4 possible solutions from E decomposition).
#                 _, R_rel, t_rel, mask = cv.recoverPose(E, pts1_inliers, pts2_inliers, K)
                
#                 # --- Step D: Image Rectification ---
#                 # We rectify the previous and current frame based on their relative motion
#                 R1_id = np.eye(3)
#                 t1_id = np.zeros((3, 1))
#                 M1, M2, K1p, K2p, R1p, R2p, t1p, t2p = rectify_pair(K, K, R1_id, R_rel, t1_id, t_rel)
                
#                 rect_prev = cv.warpPerspective(prev_gray, M1, (w_img, h_img))
#                 rect_curr = cv.warpPerspective(curr_gray, M2, (w_img, h_img))
                
#                 # --- Step E: Dense Disparity and Depth ---
#                 max_disp = 64
#                 win_size = 9
#                 dispM = get_disparity(rect_prev, rect_curr, max_disp, win_size)
#                 depthM = get_depth(dispM, K1p, K2p, R1p, R2p, t1p, t2p)
                
#                 # --- Step F: Coordinate Warping & 3D Point Extraction ---
#                 N_pts = pts2_inliers.shape[0]
#                 pts2_h = np.hstack((pts2_inliers, np.ones((N_pts, 1))))
#                 pts2_rectified_h = (M2 @ pts2_h.T).T
#                 pts2_rectified = pts2_rectified_h[:, :2] / pts2_rectified_h[:, 2, np.newaxis]
                
#                 valid_X = []
#                 valid_x_original = []
                
#                 fx_p, fy_p = K2p[0, 0], K2p[1, 1]
#                 cx_p, cy_p = K2p[0, 2], K2p[1, 2]
                
#                 for i in range(N_pts):
#                     u_rect = int(round(pts2_rectified[i, 0]))
#                     v_rect = int(round(pts2_rectified[i, 1]))

#                     if 0 <= v_rect < h_img and 0 <= u_rect < w_img:
#                         Z = depthM[v_rect, u_rect]
#                         if Z > 0 and not np.isinf(Z):
#                             # 3D point in rectified frame
#                             X_rect = (u_rect - cx_p) * Z / fx_p
#                             Y_rect = (v_rect - cy_p) * Z / fy_p
#                             point_3d_rect = np.array([X_rect, Y_rect, Z])
                            
#                             # Rotate back to original Camera 2 frame
#                             point_3d_orig = R2p.T @ point_3d_rect
                            
#                             valid_X.append(point_3d_orig)
#                             valid_x_original.append(pts2_inliers[i])

#                 valid_X = np.array(valid_X)
#                 valid_x_original = np.array(valid_x_original)

#                 # --- Step G: PnP Pose Estimation ---
#                 # Ensure we have enough points (at least 6 for DLT)
#                 if len(valid_X) >= 6:
#                     P = estimate_pose(valid_x_original, valid_X)
#                     K_est, R_pnp, t_pnp = estimate_params(P)
                    
#                     # Accumulate global trajectory
#                     global_t = global_t + global_R @ t_pnp
#                     global_R = R_pnp @ global_R
                    
#                     trajectory.append((global_R.copy(), global_t.copy()))
#                 else:
#                     print(f"Frame {frame_count}: Not enough valid depth points for PnP.")
#             else:
#                 print(f"Frame {frame_count}: Not enough inliers after 8-point RANSAC.")
#         else:
#             print(f"Frame {frame_count}: Tracking lost.")

#         # --- Step H: Replenish Features ---
#         if p1 is None or len(good_new) < 100:
#             p0 = detect_features(curr_gray)
#         else:
#             p0 = good_new.reshape(-1, 1, 2)

#         prev_gray = curr_gray.copy()
#         frame_count += 1
#         print(f"Processed frame {frame_count}/{max_frames}")

#     cap.release()
#     print("Processing complete!")

#     # ==========================================
#     # 3. VISUALIZATION
#     # ==========================================
#     # Import the visualizer given in your assignment here
#     from odometry_visualizer import plot_trajectory
#     plot_trajectory(trajectory)
#     print(f"Ready to visualize trajectory containing {len(trajectory)} poses.")

# if __name__ == "__main__":
#     main()


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Import core math functions from your submission files
from submission import essential_matrix, triangulate
from subtask2 import detect_features, track_features, camera2, refineF
from odometry_visualizer import TrajectoryVisualizer


def fast_eight_point(pts1, pts2, M, refine=True):
    """Eight point algorithm with an optional refinement toggle for speed."""
    N = pts1.shape[0]
    T = np.array([[1.0 / M, 0, 0],
                  [0, 1.0 / M, 0],
                  [0, 0, 1]])

    pts1_h = np.hstack((pts1, np.ones((N, 1))))
    pts2_h = np.hstack((pts2, np.ones((N, 1))))

    pts1_norm = (T @ pts1_h.T).T
    pts2_norm = (T @ pts2_h.T).T

    u1, v1 = pts1_norm[:, 0], pts1_norm[:, 1]
    u2, v2 = pts2_norm[:, 0], pts2_norm[:, 1]

    A = np.vstack((u2*u1, u2*v1, u2, v2*u1, v2*v1, v2, u1, v1, np.ones(N))).T

    U_a, S_a, Vt_a = np.linalg.svd(A)
    F = Vt_a[-1, :].reshape(3, 3)

    U_f, S_f, Vt_f = np.linalg.svd(F)
    S_f[-1] = 0
    F_rank2 = U_f @ np.diag(S_f) @ Vt_f
    
    # CRITICAL FIX: Only run the heavy SciPy optimizer if requested
    if refine:
        F_refined_norm = refineF(F_rank2, pts1_norm[:, :2], pts2_norm[:, :2])
    else:
        F_refined_norm = F_rank2 

    F_unnorm = T.T @ F_refined_norm @ T
    return F_unnorm

def fast_ransac(pts1, pts2, M, num_iters=1000, threshold=2.0):
    N = pts1.shape[0]
    best_F = None
    best_inlier_count = 0
    best_inliers = None
    sample_size = 8
    
    pts1_h = np.hstack((pts1, np.ones((N, 1))))
    pts2_h = np.hstack((pts2, np.ones((N, 1))))
    
    for _ in range(num_iters):
        indices = np.random.choice(N, sample_size, replace=False)
        
        # USE FAST MODE INSIDE THE LOOP (refine=False)
        F_cand = fast_eight_point(pts1[indices], pts2[indices], M, refine=False)
        
        l2 = (F_cand @ pts1_h.T).T
        l1 = (F_cand.T @ pts2_h.T).T
        
        numerator = np.abs(np.sum(pts2_h * l2, axis=1))
        norm_l2 = np.linalg.norm(l2[:, :2], axis=1)
        norm_l1 = np.linalg.norm(l1[:, :2], axis=1)
        norm_l2[norm_l2 == 0] = 1e-6
        norm_l1[norm_l1 == 0] = 1e-6

        errors = (numerator / norm_l1) + (numerator / norm_l2)
        
        inliers = np.where(errors < threshold)[0]
        if len(inliers) > best_inlier_count:
            best_inlier_count = len(inliers)
            best_F = F_cand
            best_inliers = inliers
            
    # USE HEAVY REFINEMENT ONCE AT THE VERY END (refine=True)
    if best_inliers is not None and len(best_inliers) >= sample_size:
        final_F = fast_eight_point(pts1[best_inliers], pts2[best_inliers], M, refine=True)
    else:
        final_F = best_F
        
    return final_F, best_inliers


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    video_path = 'dataset_video.mp4' 
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    K = np.array([[517.3, 0.0,   318.6],
                  [0.0,   516.5, 255.3],
                  [0.0,   0.0,   1.0  ]])
    P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))

    # Setup Live Plotting
    plt.ion() 
    visualizer = TrajectoryVisualizer()
    T_global = np.eye(4)
    visualizer.add_pose(T_global[:3, 3])

    ret, prev_frame = cap.read()
    if not ret: return
    
    M_scale = max(prev_frame.shape[:2])
    p0 = detect_features(prev_frame)

    frame_count = 1  
    max_frames = 200  

    print(f"Starting Fast Visual Odometry... Processing {max_frames} frames.")

    while cap.isOpened() and frame_count < max_frames:
        ret, curr_frame = cap.read()
        if not ret: 
            break
        
        frame_count += 1
        print(f"Processing frame {frame_count}/{max_frames} | Tracking {len(p0)} features...")

        # 1. FEATURE TRACKING
        pts1, pts2 = track_features(prev_frame, curr_frame, p0)
        
        if len(pts1) < 15:
            print("   -> Features dropped too low. Re-detecting...")
            p0 = detect_features(prev_frame)
            pts1, pts2 = track_features(prev_frame, curr_frame, p0)

        # 2. RANSAC & MOTION ESTIMATION
        if len(pts1) >= 8:
            pts1_flat = pts1.reshape(-1, 2)
            pts2_flat = pts2.reshape(-1, 2)
            
            # Use the embedded fast RANSAC function
            F, inliers = fast_ransac(pts1_flat, pts2_flat, M_scale, num_iters=1000, threshold=2.0)
            
            if inliers is not None and len(inliers) >= 8:
                pts1_good = pts1_flat[inliers]
                pts2_good = pts2_flat[inliers]

                E = essential_matrix(F, K, K)
                M2_candidates = camera2(E)
                best_M2, max_positive = None, -1
                
                # Cheirality Check
                for i in range(4):
                    M2_test = M2_candidates[:, :, i]
                    P2_test = np.dot(K, M2_test)
                    
                    pts3d = triangulate(P1, pts1_good, P2_test, pts2_good)
                    
                    pts3d_h = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
                    pts3d_c2 = (M2_test @ pts3d_h.T).T
                    
                    in_front = (pts3d[:, 2] > 0) & (pts3d_c2[:, 2] > 0)
                    num_positive = np.sum(in_front)
                    
                    if num_positive > max_positive:
                        max_positive = num_positive
                        best_M2 = M2_test

                # 3. ACCUMULATE POSE
                if best_M2 is not None:
                    T_step = np.eye(4)
                    T_step[:3, :4] = best_M2
                    
                    T_global = T_global @ np.linalg.inv(T_step)
                    
                    visualizer.add_pose(T_global[:3, 3])
                    visualizer.visualize()
                    plt.pause(0.01) # Allows the plot to update without freezing the script

        # 4. LIVE VIDEO VISUALIZATION
        display_frame = curr_frame.copy()
        for pt in pts2:
            cv.circle(display_frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
            
        cv.imshow('Monocular VO - Feature Tracking', display_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # 5. PREPARE FOR NEXT ITERATION
        if len(pts1) >= 8 and inliers is not None:
            p0 = pts2_flat[inliers].reshape(-1, 1, 2)
        else:
            p0 = detect_features(curr_frame)
            
        prev_frame = curr_frame

    print(f"Finished processing {frame_count} frames. Close the windows to exit.")
    cap.release()
    cv.destroyAllWindows()
    
    # Keep the final trajectory plot open until the user closes it manually
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()