import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

def analyze_satellite_image(image_path):
    # Load the input image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to HSV for color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define thresholds for "green" (vegetation)
    green_lower = np.array([35, 50, 50])  # Lower bound of green in HSV
    green_upper = np.array([85, 255, 255])  # Upper bound of green in HSV

    # Create a binary mask where green pixels are white and the rest are black
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)

    # Define thresholds for "soil" (brownish shades)
    soil_lower = np.array([10, 20, 20])  # Lower bound of brown in HSV
    soil_upper = np.array([25, 255, 200])  # Upper bound of brown in HSV

    # Create a binary mask for soil
    soil_mask = cv2.inRange(hsv_image, soil_lower, soil_upper)

    # Combine the green and soil masks for visualization
    combined_mask = cv2.bitwise_or(green_mask, soil_mask)

    # Find contours for green areas (trees)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize metrics
    total_tree_area = 0
    total_tree_perimeter = 0
    tree_centers = []
    total_saturation = 0
    total_hue = 0
    total_green_pixels = 0

    # Calculate metrics for each contour and saturation
    for contour in contours:
        # Calculate perimeter and area
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)

        # Get the center of the contour (centroid of tree)
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Avoid division by zero
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            tree_centers.append((cX, cY))

        # Accumulate totals
        total_tree_area += area
        total_tree_perimeter += perimeter

        # Calculate the average saturation for the current contour
        mask = np.zeros_like(green_mask)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        green_region = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)

        # Extract saturation and hue channels
        saturation_values = green_region[:, :, 1]  # Saturation channel
        hue_values = green_region[:, :, 0]  # Hue channel

        # Accumulate saturation and hue values
        total_saturation += np.sum(saturation_values)
        total_green_pixels += np.count_nonzero(saturation_values)
        total_hue += np.sum(hue_values)

    # Calculate total area of the image
    total_pixels = image.shape[0] * image.shape[1]

    # Calculate green and soil pixel counts
    green_pixels = cv2.countNonZero(green_mask)
    soil_pixels = cv2.countNonZero(soil_mask)

    # Calculate percentages
    green_percentage = (green_pixels / total_pixels) * 100
    soil_percentage = (soil_pixels / total_pixels) * 100

    # Estimation of harvest quantity (simple linear relation to surface area)
    harvest_estimation = total_tree_area / 1000  # Simple estimation

    # Tree Density (Number of trees detected)
    tree_density = len(contours)

    # Tree Spacing (Average distance between tree centers)
    if len(tree_centers) > 1:
        distances = []
        for i in range(len(tree_centers)):
            for j in range(i + 1, len(tree_centers)):
                dist = distance.euclidean(tree_centers[i], tree_centers[j])
                distances.append(dist)
        average_tree_spacing = np.mean(distances) if distances else 0
    else:
        average_tree_spacing = 0

    # Calculate average saturation (average of the non-zero saturation values)
    if total_green_pixels > 0:
        average_saturation = total_saturation / total_green_pixels
    else:
        average_saturation = 0

    # Calculate average hue (average of the non-zero hue values)
    if total_green_pixels > 0:
        average_hue = total_hue / total_green_pixels
    else:
        average_hue = 0

    # Results
    print(f"Total Tree Surface Area (in pixels): {total_tree_area}")
    print(f"Total Tree Perimeter (in pixels): {total_tree_perimeter}")
    print(f"Percentage of Green Plants: {green_percentage:.2f}%")
    print(f"Percentage of Soil: {soil_percentage:.2f}%")
    print(f"Estimated Harvest Quantity (in units): {harvest_estimation:.2f} units")
    print(f"Tree Density (Number of Trees): {tree_density}")
    print(f"Average Tree Spacing (in pixels): {average_tree_spacing:.2f} pixels")
    print(f"Average Saturation of Green Areas: {average_saturation:.2f}")
    print(f"Average Hue of Green Areas: {average_hue:.2f}")

    # Visualize results
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    
    plt.subplot(1, 3, 2)
    plt.title("Green and Soil Mask")
    plt.imshow(combined_mask, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title("Green Areas (Trees)")
    result_image = cv2.drawContours(image_rgb.copy(), contours, -1, (255, 0, 0), 2)
    plt.imshow(result_image)
    
    plt.tight_layout()
    plt.show()

image_path = r"C:\Users\HENI\Downloads\trees\trees1.jpg"
analyze_satellite_image(image_path)
