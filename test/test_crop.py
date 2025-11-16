import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def find_pear_bounding_box(image_path: str):
    """
    Find bounding box of pear objects using color detection (yellow tones).
    
    Args:
        image_path: Path to the input image
    
    Returns:
        tuple: (bounding_boxes, processed_image)
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return [], None
    
    # Convert BGR to HSV for better color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define range for yellow color (pear color)
    # Yellow hue range in HSV
    lower_yellow1 = np.array([15, 50, 50])   # Light yellow
    upper_yellow1 = np.array([35, 255, 255]) # Dark yellow
    
    # Create mask for yellow colors
    mask1 = cv2.inRange(hsv, lower_yellow1, upper_yellow1)
    
    # Also check for greenish-yellow (some pears)
    lower_yellow2 = np.array([35, 40, 40])   # Green-yellow
    upper_yellow2 = np.array([50, 255, 255]) # Yellow-green
    mask2 = cv2.inRange(hsv, lower_yellow2, upper_yellow2)
    
    # Combine masks
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Apply median blur to reduce noise
    mask = cv2.medianBlur(mask, 5)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = []
    result_img = img.copy()
    imgh, imgw = img.shape[:2]
    # Filter contours by area and aspect ratio
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter small areas
        if area > (imgh * imgw * 0.3):  # Minimum area threshold
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (pears are usually taller than wide)
            aspect_ratio = w / h
            if 0.3 <= aspect_ratio <= 2.0:  # Reasonable aspect ratio for pears
                bounding_boxes.append((x, y, x + w, y + h))
                
                # Draw bounding box on result image
                cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result_img, f'Pear {len(bounding_boxes)}', 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return bounding_boxes, result_img, mask

def test_pear_detection():
    """Test pear detection on sample images"""
    
    # Test with multiple images
    test_images = [
        "test_images/0/21_700005830334(20250924_105905)-0.jpg",
        "test_images/0/104_700005830334(20250924_111006)-0.jpg",
        "test_images/3/540_700005830339(20250924_132954)-0.jpg",
        "test_images/1/420_700005830334(20250924_131617)-0.jpg",
        "test_images/5/585_700005830334(20250924_133500)-0.jpg",
        "test_images/6/620_700005830334(20250924_133907)-0.jpg",
        "test_images/7/1136_700005830334(20250924_143648)-0.jpg"
    ]
    
    for img_path in test_images:
        if Path(img_path).exists():
            print(f"\nProcessing: {img_path}")
            
            boxes, result_img, mask = find_pear_bounding_box(img_path)
            
            print(f"Found {len(boxes)} pear(s)")
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                print(f"  Pear {i+1}: Box=({x1},{y1},{x2},{y2}), Size=({width}x{height})")
            
            # Display results
            # save images instead of showing
            cv2.imwrite(f"output_{Path(img_path).stem}_mask.png", mask)
            cv2.imwrite(f"output_{Path(img_path).stem}_result.png", result_img)
            # plt.figure(figsize=(15, 5))
            
            # # Original image
            # plt.subplot(1, 3, 1)
            # plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
            # plt.title('Original')
            # plt.axis('off')
            
            # # Mask
            # plt.subplot(1, 3, 2)
            # plt.imshow(mask, cmap='gray')
            # plt.title('Color Mask')
            # plt.axis('off')
            
            # # Result with bounding boxes
            # plt.subplot(1, 3, 3)
            # plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
            # plt.title(f'Detected Pears ({len(boxes)})')
            # plt.axis('off')
            
            # plt.tight_layout()
            # plt.show()
            
        else:
            print(f"Image not found: {img_path}")



if __name__ == "__main__":
    print("Testing pear detection with color-based approach...")
    
    # Test synthetic pears first
   
    # Test real images
    test_pear_detection()