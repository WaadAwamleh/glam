import torch
import cv2
import math
from math import sqrt
from sklearn.metrics import mean_squared_error
from skimage.feature import peak_local_max
import numpy as np
import random

def non_max_suppression(image, size_filter, proba):
    peak_coords = peak_local_max(image, min_distance=size_filter, threshold_abs=proba, exclude_border=True)
    non_max = np.zeros_like(image, dtype=bool)
    non_max[tuple(peak_coords.T)] = True
    for y, x in peak_coords:
        x_start = max(x - size_filter, 0)
        x_end = min(x + size_filter + 1, image.shape[1])
        y_start = max(y - size_filter, 0)
        y_end = min(y + size_filter + 1, image.shape[0])
        if np.sum(non_max[y_start:y_end, x_start:x_end]) > 1:
            non_max[y_start:y_end, x_start:x_end] = False
            non_max[y, x] = True
    return non_max

def sift(SIFT, image, kp_before):
    kp, des = SIFT.compute(image, kp_before)
    if des is not None:
        des += 1e-7
        des = des / des.sum(axis=1, keepdims=True)
        des = np.sqrt(des)
    return kp, des

def warp_keypoints(kp, real_H):
    num_points = kp.shape[0]
    homogeneous_points = np.concatenate([kp, np.ones((num_points, 1))], axis=1)
    warped_points = np.dot(real_H, np.transpose(homogeneous_points))
    warped_points = np.transpose(warped_points) 
    warped_points = warped_points[:, :2] / warped_points[:, 2:]
    return warped_points
    
def matching(kp1, kp2, matches, real_H, distance_threshold):

    true_wk = warp_keypoints(np.float32([kp1[m.queryIdx] for m in matches]).reshape(-1, 2), real_H)
    wk = np.float32([kp2[m.trainIdx] for m in matches]).reshape(-1, 2)
    norm = (np.linalg.norm(true_wk - wk, axis=1) <= distance_threshold)

    tp = np.sum(norm)
    fp = len(matches) - tp
    tpm = [matches[i] for i in range(norm.shape[0]) if (norm[i] == 1)]
    fpm = [matches[i] for i in range(norm.shape[0]) if (norm[i] == 0)]
    kp1_tp = np.int32([kp1[m.queryIdx] for m in tpm]).reshape(-1, 2)
    kp1_fp = np.int32([kp1[m.queryIdx] for m in fpm]).reshape(-1, 2)
    return tp, fp, tpm, kp1_tp, kp1_fp


def warp_kp(kp, homography, shape):
    warped_points = warp_keypoints(kp.T[:, [1,0]], homography)
    kp = []
    for i in range(warped_points.shape[0]):
        if ((warped_points[i, 0] >= 0) & (warped_points[i, 0] < shape[1]) &
               (warped_points[i, 1] >= 0) & (warped_points[i, 1] < shape[0])):
            kp.append(warped_points[i])
    kp = np.array(kp)
    return kp.T[[1,0], :]

def keep_shared_keypoints(kp, real_H, shape):
    warped_points = warp_keypoints(kp, real_H) 
    kp_shared = []
    for i in range(warped_points.shape[0]):
        if ((warped_points[i, 0] >= 0) & (warped_points[i, 0] < shape[1]) & \
                (warped_points[i, 1] >= 0) & (warped_points[i, 1] < shape[0])):
            kp_shared.append(kp[i])
    kp_shared = np.array(kp_shared)
    return kp_shared

def compute_repeatability(kp1, kp2, real_H, shape, distance_thresh):
    if kp1.shape[0] == 0 or kp2.shape[0] == 0:
        return 0
    warped_keypoints = keep_shared_keypoints(kp2, np.linalg.inv(real_H), shape)
    keypoints = keep_shared_keypoints(kp1, real_H, shape)
    if warped_keypoints.shape[0] == 0 or keypoints.shape[0] == 0:
        return 0
    true_warped_keypoints = warp_keypoints(keypoints, real_H)
    true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
    warped_keypoints = np.expand_dims(warped_keypoints, 0)
    distances = np.linalg.norm(true_warped_keypoints - warped_keypoints, axis=2)
    count1 = np.sum(np.min(distances, axis=1) <= distance_thresh)
    count2 = np.sum(np.min(distances, axis=0) <= distance_thresh)
    repeatability = (count1 + count2) / (keypoints.shape[0] + warped_keypoints.shape[0])
    return repeatability

    
def class_homography(MEE, MAE):
    if math.isnan(MEE) is True:
        found_homography = 0
        acceptable_homography = 0
    else:
        found_homography = 1
        if (MEE < 10 and MAE < 30):
            acceptable_homography = 1
        else:
            acceptable_homography = 0    
    return found_homography, acceptable_homography


def compute_homography(kp1, kp2, des1, des2, true_positive_matches):
    
    if des1 is not None and des2 is not None and des1.shape[0] > 2 and des2.shape[0] > 2:
    
        if len(true_positive_matches) >= 4:
            src_pts = np.float32([kp1[m.queryIdx] for m in true_positive_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx] for m in true_positive_matches]).reshape(-1, 1, 2)
            H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
            ratio_inliers = np.sum(inliers) / len(inliers) if inliers is not None else 0.0
            return H, ratio_inliers
        return np.zeros((3, 3)), 0.0  # Fallback if not enough matches or descriptors are None


def homography_is_accepted(H):
    accepted = True
    det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
    if (det < 0.2):
        accepted = False
    N1 = math.sqrt(H[0, 0] * H[0, 0] + H[1, 0] * H[1, 0])
    N2 = math.sqrt(H[0, 1] * H[0, 1] + H[1, 1] * H[1, 1])
    N3 = math.sqrt(H[2, 0] * H[2, 0] + H[2, 1] * H[
        2, 1])  

    if ((N1 > 4) or (N1 < 0.1)):
        accepted = False
    if ((N2 > 4) or (N2 < 0.1)):
        accepted = False
    if (math.fabs(N1 - N2) > 0.5):
        accepted = False
    if (N3 > 0.002):
        accepted = False

    return accepted 

def compute_registration_error(real_H, H, shape):
    if np.all(H == 0):
        return float('nan'), float('nan'), float('nan')
    corners = np.array([
        [50, 50, 1], [50, shape[0] - 50, 1],
        [shape[1] - 50, 50, 1], [shape[1] - 50, shape[0] - 50, 1],
        [shape[1] // 2, shape[0] // 4, 1],
        [shape[1] // 2, 3 * shape[0] // 4, 1]
    ])
    gt_warped_points = (real_H @ corners.T).T
    warped_points = (H @ corners.T).T
    gt_warped_points = gt_warped_points[:, :2] / gt_warped_points[:, 2, np.newaxis]
    warped_points = warped_points[:, :2] / warped_points[:, 2, np.newaxis]
    if np.any(np.isnan(warped_points)) or np.any(np.isnan(gt_warped_points)):
        return float('nan'), float('nan'), float('nan')
    try:
        RMSE = np.sqrt(mean_squared_error(warped_points, gt_warped_points))
        errors = np.linalg.norm(gt_warped_points - warped_points, axis=1)
        MEE = np.median(errors)
        MAE = np.max(errors)
        
        return RMSE, MEE, MAE
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return float('nan'), float('nan'), float('nan')
    
    
def compute_reward(image1, image2, kp_map1, kp_map2, homographies, nms, distance_threshold, device, compute_metrics=False):
    
    kp_map1, kp_map2 = kp_map1.cpu().numpy().squeeze(1), kp_map2.cpu().numpy().squeeze(1)
    image1, image2 = image1.cpu().numpy().squeeze(1), image2.cpu().numpy().squeeze(1) 
    homographies = homographies.cpu().numpy()
    
    reward_batch1 = np.zeros((image1.shape), np.float32)
    mask_batch1 = np.zeros((image1.shape), np.float32)
    
    rep_glampoints = []
    homo_glampoints = []
    acceptable_homo_glampoints = []
    kp_glampoints = []
    tp_glampoints = []
    metrics_per_image = {}
    
    SIFT = cv2.SIFT_create()
    for i in range(kp_map1.shape[0]):
        
        shape = kp_map1[i, :, :].shape
        
        plot = {}
        metrics = {}
        
        reward1 = reward_batch1[i, :, :]
        homography = homographies[i, :, :]
        
        kp_map1_nonmax = non_max_suppression(kp_map1[i, :, :], nms, 0.3)
        kp_map2_nonmax = non_max_suppression(kp_map2[i, :, :], nms, 0.3)
        
        keypoints_map1 = np.where(kp_map1_nonmax > 0)
        keypoints_map2 = np.where(kp_map2_nonmax > 0)
        
        kp1_array = np.array([keypoints_map1[1], keypoints_map1[0]]).T.astype(float)
        kp2_array = np.array([keypoints_map2[1], keypoints_map2[0]]).T.astype(float)
        
        kp1_cv2 = [cv2.KeyPoint(float(kp1_array[i, 0]), float(kp1_array[i, 1]), 10) for i in range(len(kp1_array))]
        kp2_cv2 = [cv2.KeyPoint(float(kp2_array[i, 0]), float(kp2_array[i, 1]), 10) for i in range(len(kp2_array))]
        
        kp1, des1 = sift(SIFT, np.uint8(image1[i, :, :]), kp1_cv2)
        kp2, des2 = sift(SIFT, np.uint8(image2[i, :, :]), kp2_cv2)
        
        kp1 = np.array([m.pt for m in kp1], dtype=np.int32)
        kp2 = np.array([m.pt for m in kp2], dtype=np.int32)
        
        if des1 is not None and des2 is not None:
            if des1.shape[0] > 2 and des2.shape[0] > 2:
                
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                matches1 = bf.match(des1, des2)
                tp, fp, tpm, kp1_tp, kp1_fp = matching(kp1, kp2, matches1, homography, distance_threshold=distance_threshold)
                
                reward1[kp1_tp[:, 1].tolist(), kp1_tp[:, 0].tolist()] = 1
                mask_batch1[i, kp1_tp[:, 1].tolist(), kp1_tp[:, 0].tolist()] = 1
                if tp >= fp:
                    mask_batch1[i, kp1_fp[:, 1].tolist(), kp1_fp[:, 0].tolist()] = 1.0
                else:
                    index = random.sample(range(len(kp1_fp)), tp)
                    mask_batch1[i, kp1_fp[index, 1].tolist(),
                                kp1_fp[index, 0].tolist()] = 1.0
        if compute_metrics:
            computed_H, ratio_inliers = compute_homography(kp1, kp2, des1, des2, tpm)
            tf_accepted = homography_is_accepted(computed_H)
            RMSE, MEE, MAE = compute_registration_error(homography, computed_H, shape)
            found_homography, acceptable_homography = class_homography(MEE, MAE)
            metrics['computed_H'] = computed_H
            metrics['homography_correct'] = tf_accepted
            metrics['inlier_ratio'] = ratio_inliers
            metrics['class_acceptable'] = acceptable_homography
            
            if (kp1.shape[0] != 0) and (kp2.shape[0] != 0):
                repeatability = compute_repeatability(kp1, kp2, homography, shape,distance_threshold)
            else:
                repeatability = 0
            metrics['repeatability'] = repeatability
            plot['keypoints_map1'] = keypoints_map1
            plot['keypoints_map2'] = keypoints_map2
            metrics['nbr_kp1'] = len(keypoints_map1[0])
            metrics['nbr_kp2'] = len(keypoints_map2[0])
            tp_kp1 = kp1_tp.T[[1,0], :]
            if len(tp_kp1[1]) != 0:
                where_warped_tp_kp1 = warp_kp(tp_kp1, homography, (kp_map1.shape[1], kp_map1.shape[2]))
            else:
                where_warped_tp_kp1 = np.zeros((2, 1))
            plot['tp_kp1'] = tp_kp1
            plot['warped_tp_kp1'] = where_warped_tp_kp1
            metrics['total_nbr_kp_reward1'] = np.sum(reward1)
            metrics['to_plot'] = plot
            metrics_per_image['{}'.format(i)] = metrics
            rep_glampoints.append(repeatability)
            homo_glampoints.append(tf_accepted)
            acceptable_homo_glampoints.append(acceptable_homography)
            kp_glampoints.append(metrics['nbr_kp1'])
            tp_glampoints.append(metrics['total_nbr_kp_reward1'])
    reward_batch1 = torch.from_numpy(reward_batch1).unsqueeze(1).to(device)
    mask_batch1 = torch.from_numpy(mask_batch1).unsqueeze(1).to(device)
    del SIFT
    if compute_metrics:
        metrics_per_image['sum_rep'] = np.sum(rep_glampoints)
        metrics_per_image['nbr_homo_correct'] = np.sum(homo_glampoints)
        metrics_per_image['nbr_homo_acceptable'] = np.sum(acceptable_homography)
        metrics_per_image['nbr_kp'] = np.sum(kp_glampoints)
        metrics_per_image['nbr_tp'] = np.sum(tp_glampoints)
        metrics_per_image['nbr_images'] = len(tp_glampoints)
        return reward_batch1, mask_batch1, metrics_per_image
    else:
        return reward_batch1, mask_batch1
        

def compute_loss(reward, kpmap, mask):
    loss_matrix = (reward - kpmap)**2 * mask
    loss = loss_matrix / (torch.sum(mask, axis=[1,2,3]).unsqueeze(1).unsqueeze(1).unsqueeze(1)+1e-6)
    return loss.sum()