
import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Constants for the new robust matching logic
LOWE_RATIO = 0.75
MIN_PRIMARY_INLIERS = 12
MIN_SECONDARY_INLIERS_CONSISTENT = 10 # Min number of secondary points that must agree with the primary transform
SECONDARY_VERIFICATION_THRESHOLD_PX = 7.5 # Max distance (in pixels) for a secondary point to be considered consistent

def identify_vendor_via_features(
    processed_image_obj: Image.Image, 
    orb_templates: list
) -> tuple[str | None, dict | None, tuple[np.ndarray, np.ndarray] | None]:
    """
    Identifies a vendor using a context-invariant feature matching strategy.
    
    This definitive algorithm works by:
    1.  Generating a single, global set of keypoints and descriptors from the new document.
    2.  For each template, it first finds and validates the primary anchor against this global set.
    3.  If the primary anchor is found, it uses the resulting geometric transformation to verify
        that the secondary anchor's features also exist and are in the correct relative positions.
    This two-factor geometric lock prevents context-switching errors and provides maximum stability.
    """
    if not orb_templates: return None, None, None

    logger.debug("Starting context-invariant feature matching.")
    incoming_cv_image = np.array(processed_image_obj)

    # Step 1: Create a single, authoritative context for the incoming document.
    orb = cv2.ORB_create(nfeatures=5000)
    kp_incoming, des_incoming = orb.detectAndCompute(incoming_cv_image, None)
    
    if des_incoming is None:
        logger.warning("Could not find any features in the incoming document.")
        return None, None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    for template in orb_templates:
        vendor_name = template["vendor_name"]
        
        # --- Stage 1: Find and Validate the Primary Anchor ---
        primary_des_template = template["primary_anchor"]["anchor_descriptors"]
        primary_kp_template = template["primary_anchor"]["anchor_keypoints_pts"]
        
        # Match primary anchor against the global descriptor set
        matches_primary = bf.knnMatch(primary_des_template, des_incoming, k=2)
        
        good_matches_primary = []
        for match_pair in matches_primary:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < LOWE_RATIO * n.distance:
                    good_matches_primary.append(m)

        if len(good_matches_primary) < MIN_PRIMARY_INLIERS:
            continue # Not enough initial matches, try next template

        # Get the coordinates for all good matches
        src_pts_primary = primary_kp_template[[m.queryIdx for m in good_matches_primary]]
        dst_pts_primary = np.float32([kp_incoming[m.trainIdx].pt for m in good_matches_primary])
        
        # Validate the primary match with a robust affine model
        affine_matrix, inlier_mask_primary = cv2.estimateAffinePartial2D(src_pts_primary, dst_pts_primary, method=cv2.RANSAC)
        
        if affine_matrix is None or np.sum(inlier_mask_primary) < MIN_PRIMARY_INLIERS:
            continue # Geometric validation failed for primary anchor

        logger.info(f"Primary anchor candidate CONFIRMED for '{vendor_name}'. Verifying secondary anchor...")
        
        # --- Stage 2: Verify the Secondary Anchor Using the Primary Transform ---
        secondary_des_template = template["secondary_anchor"]["anchor_descriptors"]
        secondary_kp_template = template["secondary_anchor"]["anchor_keypoints_pts"]
        
        # Match secondary anchor against the SAME global descriptor set
        matches_secondary = bf.knnMatch(secondary_des_template, des_incoming, k=2)
        
        good_matches_secondary = []
        for match_pair in matches_secondary:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < LOWE_RATIO * n.distance:
                    good_matches_secondary.append(m)

        if len(good_matches_secondary) < MIN_SECONDARY_INLIERS_CONSISTENT:
            continue # Not enough secondary matches to even attempt verification
            
        # Get coordinates for secondary matches
        src_pts_secondary = secondary_kp_template[[m.queryIdx for m in good_matches_secondary]]
        dst_pts_secondary_actual = np.float32([kp_incoming[m.trainIdx].pt for m in good_matches_secondary])

        # Use the transform from the primary anchor to predict where the secondary points should be
        # Note: cv2.transform needs shape (N, 1, 2)
        src_pts_secondary_reshaped = src_pts_secondary.reshape(-1, 1, 2)
        dst_pts_secondary_predicted = cv2.transform(src_pts_secondary_reshaped, affine_matrix)
        
        # Calculate the distance between actual and predicted points
        distances = np.linalg.norm(dst_pts_secondary_actual.reshape(-1, 1, 2) - dst_pts_secondary_predicted, axis=2)
        
        # Check how many points are consistent with the primary transformation
        consistent_point_count = np.sum(distances < SECONDARY_VERIFICATION_THRESHOLD_PX)
        
        if consistent_point_count >= MIN_SECONDARY_INLIERS_CONSISTENT:
            logger.info(f"Secondary anchor CONFIRMED for '{vendor_name}'.")
            logger.info(f"GEOMETRIC LOCK CONFIRMED! Best match is '{vendor_name}'.")

            # Success! Combine the inliers from the primary match for the final transformation
            primary_src_inliers = src_pts_primary[inlier_mask_primary.ravel() == 1]
            primary_dst_inliers = dst_pts_primary[inlier_mask_primary.ravel() == 1]
            
            # For simplicity in this fix, we will use only the primary inliers for transformation.
            # A more advanced system could combine both, but this is robust.
            
            return (
                vendor_name, 
                template["full_template_data"], 
                (primary_src_inliers, primary_dst_inliers)
            )
            
    logger.info("Could not identify vendor via Hybrid Regional Search.")
    return None, None, None