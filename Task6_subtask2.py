import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Import core math functions from your submission files
from submission import essential_matrix, triangulate
from subtask2 import detect_features, track_features, camera2, refineF
from odometry_visualizer import TrajectoryVisualizer


def fast_eight_point(pts1, pts2, M, refine=True):
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


# MAIN PIPELINE
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
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
