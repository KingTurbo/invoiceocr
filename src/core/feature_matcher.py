import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MIN_MATCH_COUNT = 15 # Can be slightly more lenient as we need two matches
LOWE_RATIO = 0.75

def _find_single_anchor(
    template_anchor_data: dict, 
    kp_incoming: list, 
    des_incoming: np.ndarray, 
    bf_matcher: cv2.BFMatcher
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Attempts to find a single anchor and returns the matched keypoints."""
    des_template = template_anchor_data["anchor_descriptors"]
    kp_template_pts = template_anchor_data["anchor_keypoints_pts"]
    
    matches = bf_matcher.knnMatch(des_template, des_incoming, k=2)

    good_matches = []
    try:
        for m, n in matches:
            if m.distance < LOWE_RATIO * n.distance:
                good_matches.append(m)
    except ValueError:
        return None, None

    if len(good_matches) >= MIN_MATCH_COUNT:
        # Get the coordinates of matched keypoints from both the template (source) and the incoming image (destination)
        src_pts = np.float32([kp_template_pts[m.queryIdx] for m in good_matches])
        dst_pts = np.float32([kp_incoming[m.trainIdx].pt for m in good_matches])
        return src_pts, dst_pts
        
    return None, None

def identify_vendor_via_features(
    processed_image_obj: Image.Image, 
    orb_templates: list
) -> tuple[str | None, dict | None, tuple[np.ndarray, np.ndarray] | None]:
    """
    Identifies a vendor by finding BOTH the primary and secondary anchors.
    Returns the vendor name, template, and the source/destination points
    for the affine transformation.
    """
    if not orb_templates:
        return None, None, None

    logger.debug(f"Starting 2-Anchor feature matching against {len(orb_templates)} templates.")
    
    try:
        incoming_cv_image = np.array(processed_image_obj)
        orb = cv2.ORB_create(nfeatures=4000) # Increased features for better whole-page matching
        kp_incoming, des_incoming = orb.detectAndCompute(incoming_cv_image, None)

        if des_incoming is None or len(des_incoming) < MIN_MATCH_COUNT * 2:
            logger.warning("Not enough features on incoming document to attempt matching.")
            return None, None, None
    except Exception as e:
        logger.error(f"Failed to extract features from incoming document. Error: {e}")
        return None, None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    for template in orb_templates:
        vendor_name = template["vendor_name"]
        
        # --- Step 1: Find the Primary Anchor ---
        primary_src_pts, primary_dst_pts = _find_single_anchor(
            template["primary_anchor"], kp_incoming, des_incoming, bf
        )
        if primary_src_pts is None:
            # logger.debug(f"Primary anchor for '{vendor_name}' not found. Skipping.")
            continue
        
        logger.info(f"Primary anchor candidate found for '{vendor_name}' with {len(primary_src_pts)} matches. Searching for secondary...")

        # --- Step 2: Find the Secondary Anchor ---
        secondary_src_pts, secondary_dst_pts = _find_single_anchor(
            template["secondary_anchor"], kp_incoming, des_incoming, bf
        )
        if secondary_src_pts is None:
            # logger.debug(f"Secondary anchor for '{vendor_name}' not found. Aborting candidate.")
            continue

        logger.info(f"Secondary anchor CONFIRMED for '{vendor_name}' with {len(secondary_src_pts)} matches.")
        
        # --- Success! Both anchors were found ---
        logger.info(f"GEOMETRIC LOCK CONFIRMED! Best match is '{vendor_name}'.")

        # Combine the points from both anchors to create a robust set for transformation
        all_src_pts = np.vstack((primary_src_pts, secondary_src_pts))
        all_dst_pts = np.vstack((primary_dst_pts, secondary_dst_pts))

        return (
            vendor_name, 
            template["full_template_data"], 
            (all_src_pts, all_dst_pts)
        )
        
    logger.info("Could not identify vendor via 2-anchor feature matching.")
    return None, None, None