import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MIN_MATCH_COUNT = 25
LOWE_RATIO = 0.75

def identify_vendor_via_features(
    processed_image_obj: Image.Image, 
    orb_templates: list
) -> tuple[str | None, dict | None, np.ndarray | None]:
    if not orb_templates:
        return None, None, None

    logger.debug(f"Starting feature matching against {len(orb_templates)} ORB templates.")
    
    try:
        incoming_cv_image = np.array(processed_image_obj)
        orb = cv2.ORB_create(nfeatures=2000)
        kp_incoming, des_incoming = orb.detectAndCompute(incoming_cv_image, None)

        if des_incoming is None or len(des_incoming) < MIN_MATCH_COUNT:
            logger.warning("Not enough features on incoming document to attempt matching.")
            return None, None, None
    except Exception as e:
        logger.error(f"Failed to extract features from incoming document. Error: {e}")
        return None, None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    best_match_info = {
        "vendor_name": None,
        "template_data": None,
        "anchor_bbox": None,
        "max_good_matches": 0
    }

    for template in orb_templates:
        vendor_name = template["vendor_name"]
        des_template = template["anchor_descriptors"]
        
        matches = bf.knnMatch(des_template, des_incoming, k=2)

        good_matches = []
        try:
            for m, n in matches:
                if m.distance < LOWE_RATIO * n.distance:
                    good_matches.append(m)
        except ValueError:
            continue
        
        if len(good_matches) > best_match_info["max_good_matches"] and len(good_matches) >= MIN_MATCH_COUNT:
            # --- FIX: Use the REAL keypoint coordinates from the template ---
            # Get the stored list of (x,y) coordinates from the cached template
            kp_template_pts = template["anchor_keypoints_pts"]

            # The original flawed line is removed:
            # kp_template = [cv2.KeyPoint(0, 0, 1)] * len(des_template) # <-- REMOVED

            # Use the real coordinates to build the source points array
            src_pts = np.float32([kp_template_pts[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_incoming[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            # --- END FIX ---

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                h, w = template["anchor_source_shape"]['height'], template["anchor_source_shape"]['width']
                src_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                
                try:
                    dst_corners = cv2.perspectiveTransform(src_corners, M)
                    best_match_info["vendor_name"] = vendor_name
                    best_match_info["template_data"] = template["full_template_data"]
                    best_match_info["anchor_bbox"] = np.int32(dst_corners)
                    best_match_info["max_good_matches"] = len(good_matches)
                except cv2.error:
                    logger.warning(f"Perspective transform failed for '{vendor_name}'.")
                    continue
    
    if best_match_info["vendor_name"]:
        vendor = best_match_info["vendor_name"]
        matches = best_match_info["max_good_matches"]
        logger.info(f"FEATURE MATCH CONFIRMED! Best match is '{vendor}' with {matches} good matches.")
        return (best_match_info["vendor_name"], best_match_info["template_data"], best_match_info["anchor_bbox"])
        
    logger.info("Could not identify vendor via feature matching.")
    return None, None, None