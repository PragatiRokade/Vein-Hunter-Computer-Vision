import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import frangi

def process_veins_live_tuner(image_path):
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
    print("Draw a box strictly inside your hand (no background). Press SPACE to confirm.")
    bbox = cv2.selectROI("1. Select Target Area", img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("1. Select Target Area")
    x, y, w, h = bbox
    
    if w == 0 or h == 0:
        print("Selection failed. Exiting.")
        return

    roi_mask = np.zeros((new_height, new_width), dtype=np.uint8)
    cv2.rectangle(roi_mask, (x, y), (x+w, y+h), 255, -1)

    # --- 3. AGGRESSIVE PREPROCESSING ---
    b, green_channel, r = cv2.split(img)
    
    # Cranked up the CLAHE limit to 3.0 to force more contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_green = clahe.apply(green_channel)
    
    # A MUCH stronger blur (15x15) to completely melt skin pores/hair, leaving only broad veins
    blurred_green = cv2.GaussianBlur(enhanced_green, (15, 15), 0)

    # --- 4. HEAVY MATH (Calculated Once) ---
    print("Calculating Frangi filter... please wait.")
    # Tuned sigmas for thicker, blurred structures
    vein_probabilities = frangi(blurred_green, sigmas=np.arange(3, 10, 2), black_ridges=True)
    vein_probabilities[roi_mask == 0] = 0
    vein_prob_normalized = cv2.normalize(vein_probabilities, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # --- 5. THE LIVE TUNING GUI ---
    print("Opening Live Tuner. Adjust the slider until the veins connect into solid lines.")
    print("Press 'q' or 'ENTER' on your keyboard to finalize and view the report.")
    
    cv2.namedWindow("2. Live Vein Tuner")
    
    # This function runs every time you move the slider
    def on_trackbar(val):
        # 1. Apply the current threshold from the slider
        _, binary_veins = cv2.threshold(vein_prob_normalized, val, 255, cv2.THRESH_BINARY)
        
        # 2. Use Morphological Dilation to connect the "dots" into solid tubes
        kernel = np.ones((5,5), np.uint8)
        connected_veins = cv2.morphologyEx(binary_veins, cv2.MORPH_CLOSE, kernel)
        
        # 3. Create the green overlay
        live_overlay = img.copy()
        live_overlay[connected_veins == 255] = [0, 255, 0]
        
        # Show the result in the window
        cv2.imshow("2. Live Vein Tuner", live_overlay)
        return connected_veins

    # Create the slider (starts at 5, goes up to 100)
    cv2.createTrackbar("Threshold", "2. Live Vein Tuner", 5, 100, on_trackbar)
    
    # Force the first frame to draw
    final_veins = on_trackbar(5)

    # Wait for the user to press 'q' or 'Enter'
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 13: # 13 is Enter key
            break
            
    # Grab the final value from the slider before closing
    final_threshold = cv2.getTrackbarPos("Threshold", "2. Live Vein Tuner")
    final_veins = on_trackbar(final_threshold)
    cv2.destroyAllWindows()

    # --- 6. FINAL VISUALIZATION ---
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.title("1. Heavily Blurred Green Channel")
    plt.imshow(blurred_green, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("2. Frangi Map")
    plt.imshow(vein_prob_normalized, cmap='hot')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title(f"3. Connected Vein Mask (Threshold: {final_threshold})")
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

# Run the pipeline
process_veins_live_tuner('hand.jpeg')