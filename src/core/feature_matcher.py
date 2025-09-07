import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MIN_INLIER_COUNT = 12
LOWE_RATIO = 0.75
PRIMARY_ANCHOR_SEARCH_REGION = (0.0, 0.0, 1.0, 0.35) 

def _find_anchor_in_large_region(
    incoming_cv_image: np.ndarray,
    template_anchor_data: dict
) -> tuple[np.ndarray | None, np.ndarray | None]:
    h, w = incoming_cv_image.shape[:2]
    x_start, y_start, x_end_rel, y_end_rel = PRIMARY_ANCHOR_SEARCH_REGION
    
    search_area_y_end = int(h * y_end_rel)
    search_area_x_end = int(w * x_end_rel)
    search_region_cv = incoming_cv_image[0:search_area_y_end, 0:search_area_x_end]

    orb = cv2.ORB_create(nfeatures=4000)
    kp_incoming, des_incoming = orb.detectAndCompute(search_region_cv, None)

    if des_incoming is None:
        return None, None
        
    des_template = template_anchor_data["anchor_descriptors"]
    kp_template_pts = template_anchor_data["anchor_keypoints_pts"]
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des_template, des_incoming, k=2)

    # --- THIS IS THE CORRECTED LOGIC ---
    good_matches = []
    for match_pair in matches:
        # Check if knnMatch returned a pair of matches
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < LOWE_RATIO * n.distance:
                good_matches.append(m)
    # --- END CORRECTION ---

    if len(good_matches) >= MIN_INLIER_COUNT:
        src_pts_all = np.float32([kp_template_pts[m.queryIdx] for m in good_matches])
        dst_pts_all = np.float32([kp_incoming[m.trainIdx].pt for m in good_matches])

        M, mask = cv2.findHomography(src_pts_all, dst_pts_all, cv2.RANSAC, 5.0)
        
        if M is not None and np.sum(mask) >= MIN_INLIER_COUNT:
            src_pts_inliers = src_pts_all[mask.ravel() == 1]
            dst_pts_inliers = dst_pts_all[mask.ravel() == 1]
            return src_pts_inliers, dst_pts_inliers
            
    return None, None

def _find_anchor_in_guided_roi(
    incoming_cv_image: np.ndarray,
    template_anchor_data: dict,
    predicted_center: tuple[int, int]
) -> tuple[np.ndarray | None, np.ndarray | None]:
    roi_size = 200
    px, py = int(predicted_center[0]), int(predicted_center[1])

    y_start = max(0, py - roi_size // 2)
    y_end = py + roi_size // 2
    x_start = max(0, px - roi_size // 2)
    x_end = px + roi_size // 2
    
    roi_crop_cv = incoming_cv_image[y_start:y_end, x_start:x_end]
    
    if roi_crop_cv.size == 0: return None, None

    orb = cv2.ORB_create(nfeatures=1500)
    kp_incoming_roi, des_incoming_roi = orb.detectAndCompute(roi_crop_cv, None)

    if des_incoming_roi is None: return None, None
        
    des_template = template_anchor_data["anchor_descriptors"]
    kp_template_pts = template_anchor_data["anchor_keypoints_pts"]
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des_template, des_incoming_roi, k=2)

    # --- THIS IS THE SAME CORRECTION, APPLIED HERE AS WELL ---
    good_matches = []
    for match_pair in matches:
        # Check if knnMatch returned a pair of matches
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < LOWE_RATIO * n.distance:
                good_matches.append(m)
    # --- END CORRECTION ---

    if len(good_matches) >= MIN_INLIER_COUNT:
        src_pts_all = np.float32([kp_template_pts[m.queryIdx] for m in good_matches])
        dst_pts_all = np.float32([(kp.pt[0] + x_start, kp.pt[1] + y_start) for kp in [kp_incoming_roi[m.trainIdx] for m in good_matches]])

        M, mask = cv2.findHomography(src_pts_all, dst_pts_all, cv2.RANSAC, 5.0)
        
        if M is not None and np.sum(mask) >= MIN_INLIER_COUNT:
            src_pts_inliers = src_pts_all[mask.ravel() == 1]
            dst_pts_inliers = dst_pts_all[mask.ravel() == 1]
            return src_pts_inliers, dst_pts_inliers
            
    return None, None

def identify_vendor_via_features(
    processed_image_obj: Image.Image, 
    orb_templates: list
) -> tuple[str | None, dict | None, tuple[np.ndarray, np.ndarray] | None]:
    if not orb_templates: return None, None, None

    logger.debug(f"Starting Hybrid Regional Search against {len(orb_templates)} templates.")
    incoming_cv_image = np.array(processed_image_obj)

    for template in orb_templates:
        vendor_name = template["vendor_name"]
        
        primary_src_pts, primary_dst_pts = _find_anchor_in_large_region(
            incoming_cv_image, template["primary_anchor"]
        )
        if primary_src_pts is None:
            continue
        
        logger.info(f"Primary anchor candidate found for '{vendor_name}'. Calculating relative secondary position...")

        p_box = template["primary_anchor"]["bounding_box"]
        s_box = template["secondary_anchor"]["bounding_box"]
        p_center_template = (p_box['x'] + p_box['width']/2, p_box['y'] + p_box['height']/2)
        s_center_template = (s_box['x'] + s_box['width']/2, s_box['y'] + s_box['height']/2)
        vector = (s_center_template[0] - p_center_template[0], s_center_template[1] - p_center_template[1])

        p_center_found = np.mean(primary_dst_pts, axis=0)
        
        predicted_s_center = (p_center_found[0] + vector[0], p_center_found[1] + vector[1])

        secondary_src_pts, secondary_dst_pts = _find_anchor_in_guided_roi(
            incoming_cv_image, template["secondary_anchor"], predicted_s_center
        )

        if secondary_src_pts is None:
            continue

        logger.info(f"Secondary anchor CONFIRMED for '{vendor_name}'.")
        logger.info(f"GEOMETRIC LOCK CONFIRMED! Best match is '{vendor_name}'.")

        all_src_pts = np.vstack((primary_src_pts, secondary_src_pts))
        all_dst_pts = np.vstack((primary_dst_pts, secondary_dst_pts))

        return (
            vendor_name, 
            template["full_template_data"], 
            (all_src_pts, all_dst_pts)
        )
        
    logger.info("Could not identify vendor via Hybrid Regional Search.")
    return None, None, None