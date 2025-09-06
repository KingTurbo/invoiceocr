import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# The minimum number of high-quality feature matches required to confirm an identification.
# This value is critical for balancing accuracy against tolerance for variations.
MIN_MATCH_COUNT = 25 
LOWE_RATIO = 0.75 # Standard value for Lowe's ratio test for filtering matches.

def identify_vendor_via_features(
    processed_image_obj: Image.Image, 
    orb_templates: list
) -> tuple[str | None, dict | None]:
    """
    Identifies a vendor by matching ORB features from an incoming document
    against a cache of pre-computed template features.

    This is the core of the new, robust identification system. It is designed
    to be resilient to changes in scale and rotation.

    Args:
        processed_image_obj: The preprocessed (de-skewed) PIL image of the
                             incoming document.
        orb_templates: A list of cached templates of type 'orb_features'.

    Returns:
        A tuple containing the identified vendor name and the full template data
        if a confident match is found; otherwise, returns (None, None).
    """
    if not orb_templates:
        return None, None

    logger.debug(f"Starting feature matching against {len(orb_templates)} ORB templates.")
    
    # 1. Prepare the incoming image and extract its features just once.
    try:
        incoming_cv_image = np.array(processed_image_obj)
        orb = cv2.ORB_create(nfeatures=2000) # Increase features for better matching potential
        kp_incoming, des_incoming = orb.detectAndCompute(incoming_cv_image, None)

        if des_incoming is None or len(des_incoming) < MIN_MATCH_COUNT:
            logger.warning("Not enough features could be detected on the incoming document to attempt matching.")
            return None, None
    except Exception as e:
        logger.error(f"Failed to extract features from incoming document. Error: {e}")
        return None, None

    # 2. Iterate through cached templates and find the best match.
    # Use a Brute-Force Matcher optimized for ORB's binary descriptors.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    best_match_vendor = None
    best_match_template = None
    max_good_matches = 0

    for template in orb_templates:
        vendor_name = template["vendor_name"]
        des_template = template["anchor_descriptors"]
        
        # Find potential matches between template and incoming document descriptors.
        matches = bf.knnMatch(des_template, des_incoming, k=2)

        # Apply Lowe's ratio test to filter out weak/ambiguous matches.
        good_matches = []
        try:
            for m, n in matches:
                if m.distance < LOWE_RATIO * n.distance:
                    good_matches.append(m)
        except ValueError:
            # This can happen if k=2 finds less than 2 matches.
            logger.debug(f"Not enough knn matches for '{vendor_name}' to perform ratio test.")
            continue
        
        logger.debug(f"Vendor '{vendor_name}' produced {len(good_matches)} good matches.")

        # Check if this is the best candidate so far.
        if len(good_matches) > max_good_matches and len(good_matches) >= MIN_MATCH_COUNT:
            max_good_matches = len(good_matches)
            best_match_vendor = vendor_name
            best_match_template = template["full_template_data"]

    if best_match_vendor:
        logger.info(f"FEATURE MATCH CONFIRMED! Best match is '{best_match_vendor}' with {max_good_matches} good matches.")
        return best_match_vendor, best_match_template
        
    logger.info("Could not identify vendor via feature matching. No template met the minimum match count.")
    return None, None