import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

def correct_skew(image_obj: Image.Image) -> Image.Image:
    """
    Corrects the skew of a document image using a high-performance,
    Hough Line Transform-based algorithm.

    This function is a critical, performance-optimized first step in the
    preprocessing pipeline. It rotates the input PIL Image to ensure that text
    lines are perfectly horizontal.

    The revised high-performance algorithm operates by:
    1. Converting the PIL image to an OpenCV-compatible NumPy array.
    2. Downscaling the image to a fixed height for rapid processing,
       significantly reducing the number of pixels for analysis.
    3. Applying a robust binarization (Otsu's method) and edge detection
       (Canny) to isolate the structural lines of the text.
    4. Using the Hough Line Transform on the edge-detected image to find all
       potential lines. This is a highly efficient, standard computer vision
       technique for this task.
    5. Calculating the median angle of the detected lines, which provides a
       highly robust estimate of the document's overall skew, ignoring outliers.
    6. Calculating the rotation matrix for the determined median angle and
       applying it to the original, full-resolution image in a single,
       hardware-accelerated operation using cv2.warpAffine.

    This approach is orders of magnitude faster than the previous projection
    profile method as it avoids costly iterative image rotations.

    Args:
        image_obj: The input PIL Image object (expected to be grayscale).

    Returns:
        A new PIL Image object with the skew corrected. Returns the original
        image if the input is invalid or skew correction fails.
    """
    logger.debug("Initiating high-performance skew correction...")
    try:
        # Convert PIL image to an OpenCV-compatible NumPy array
        original_cv = np.array(image_obj)

        # --- PERFORMANCE OPTIMIZATION 1: Downscale for detection ---
        # We determine the angle on a smaller image for massive speed gains.
        h, w = original_cv.shape
        # Use a slightly smaller height to speed up Canny and Hough even more
        target_h = 600
        scale = target_h / h
        resized_cv = cv2.resize(original_cv, (int(w * scale), target_h))
        
        # --- NEW: Get dimensions of the scaled image for relative parameter tuning ---
        rh, rw = resized_cv.shape

        # --- ALGORITHM RE-ARCHITECTURE: Hough Transform ---
        # Binarize and find edges for line detection.
        _, binary_img = cv2.threshold(resized_cv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edges = cv2.Canny(binary_img, 50, 150, apertureSize=3)

        # --- MODIFICATION: Make Hough Transform parameters more forgiving ---
        # Lowering the threshold makes it easier to detect lines.
        # Reducing minLineLength allows it to detect shorter text segments.
        # Increasing maxLineGap helps connect words on the same line.
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi / 180, 
            threshold=80,  # Lowered from 100
            minLineLength=rw // 4, # Changed from w/3 to a more relative and smaller rw/4
            maxLineGap=25 # Increased from 20
        )
        # --- END MODIFICATION ---

        if lines is None:
            logger.warning("No lines detected for skew correction. Returning original image.")
            return image_obj

        # Calculate the angle for each detected line and filter for near-horizontal lines.
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
            if -45 < angle < 45: # Focus on relevant skew angles
                angles.append(angle)

        if not angles:
            logger.warning("No suitable lines found for skew angle calculation. Returning original image.")
            return image_obj

        # Use the median angle for robust outlier rejection.
        median_angle = np.median(angles)
        
        # --- NEW: Prevent over-correction of nearly straight images ---
        if abs(median_angle) < 0.05:
            logger.info("Image skew is negligible. No correction applied.")
            return image_obj
        # --- END NEW ---

        logger.info(f"High-performance skew correction complete. Optimal angle determined: {median_angle:.2f} degrees.")
        
        # --- PERFORMANCE OPTIMIZATION 2: Single, Accelerated Rotation ---
        # Apply the rotation to the ORIGINAL, full-resolution image.
        (h, w) = original_cv.shape
        center = (w // 2, h // 2)
        
        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        # Perform the affine transformation (rotation)
        rotated_cv = cv2.warpAffine(original_cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255) # 255 for white

        # Convert the corrected OpenCV image back to a PIL image
        return Image.fromarray(rotated_cv)

    except Exception as e:
        logger.error(f"An unexpected error occurred during high-performance skew correction. Error: {e}", exc_info=True)
        return image_obj