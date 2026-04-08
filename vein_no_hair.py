import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi

def process_veins_final(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from '{image_path}'.")
        return

    # --- 1. STANDARDIZE RESOLUTION ---
    height, width = img.shape[:2]
    new_width = 600
    new_height = int((new_width / width) * height)
    img = cv2.resize(img, (new_width, new_height))

    # --- 2. INTERACTIVE ROI SELECTION ---
    print("Draw a box strictly inside your hand. Press SPACE to confirm.")
    bbox = cv2.selectROI("1. Select Target Area", img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("1. Select Target Area")
    x, y, w, h = bbox
    
    if w == 0 or h == 0:
        print("Selection failed. Exiting.")
        return

    roi_mask = np.zeros((new_height, new_width), dtype=np.uint8)
    cv2.rectangle(roi_mask, (x, y), (x+w, y+h), 255, -1)

    # --- 3. PREPROCESSING ---
    b, green_channel, r = cv2.split(img)
    
    # Very heavy blur first to kill the skin pores before contrast enhancement
    blurred_green = cv2.GaussianBlur(green_channel, (11, 11), 0)
    
    # Moderate contrast boost
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_green = clahe.apply(blurred_green)

    # --- 4. FRANGI FILTER ---
    print("Calculating Frangi filter... please wait.")
    vein_probabilities = frangi(enhanced_green, sigmas=np.arange(3, 11, 2), black_ridges=True)
    vein_probabilities[roi_mask == 0] = 0
    vein_prob_normalized = cv2.normalize(vein_probabilities, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # --- 5. THE LIVE TUNER WITH CONTOUR FILTERING ---
    cv2.namedWindow("2. Live Vein Tuner")
    
    def on_trackbar(val):
        # 1. Standard Threshold
        _, binary_veins = cv2.threshold(vein_prob_normalized, val, 255, cv2.THRESH_BINARY)
        
        # 2. NEW: Contour Area Filtering (Kill the speckles!)
        # Find all distinct shapes in the binary mask
        contours, _ = cv2.findContours(binary_veins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a blank black canvas to draw only the BIG shapes onto
        clean_mask = np.zeros_like(binary_veins)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # If the shape is larger than 30 pixels, it's a vein. Keep it.
            # If it's smaller, it's a pore/hair. Ignore it.
            if area > 30: 
                cv2.drawContours(clean_mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # 3. Aggressive Bridging on the surviving big shapes
        bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        connected_veins = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, bridge_kernel)
        
        # 4. Overlay
        live_overlay = img.copy()
        live_overlay[connected_veins == 255] = [0, 255, 0]
        
        cv2.imshow("2. Live Vein Tuner", live_overlay)
        return connected_veins

    cv2.createTrackbar("Threshold", "2. Live Vein Tuner", 10, 100, on_trackbar)
    on_trackbar(10)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13: # Enter key
            break
            
    final_threshold = cv2.getTrackbarPos("Threshold", "2. Live Vein Tuner")
    final_veins = on_trackbar(final_threshold)
    cv2.destroyAllWindows()

    # --- 6. FINAL VISUALIZATION ---
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.title("1. Blurred & CLAHE Green")
    plt.imshow(enhanced_green, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("2. Frangi Map")
    plt.imshow(vein_prob_normalized, cmap='hot')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title(f"3. Filtered Vein Mask (Threshold: {final_threshold})")
    plt.imshow(final_veins, cmap='gray')
    plt.axis('off')

    overlay = img.copy()
    overlay[final_veins == 255] = [0, 255, 0] 
    
    plt.subplot(2, 2, 4)
    plt.title("4. Final Result")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

process_veins_final('hand.jpeg')