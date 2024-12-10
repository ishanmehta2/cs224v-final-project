import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import os
from PIL import Image
from io import BytesIO

def extract_text_with_coordinates(image_path, validation_function=None):
    # Load the image
    img = cv2.imread(image_path)

    # Use Tesseract to extract data
    d = pytesseract.image_to_data(img, output_type=Output.DICT)

    # Initialize variables to store results
    extracted_text_with_coords = []

    # Extract text and bounding box coordinates
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        text = d['text'][i].strip()
        if text:  # Only include non-empty text
            x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
            bounding_box = (x, y, x + w, y + h)
            extracted_text_with_coords.append((text, bounding_box))

            # Optionally validate each bit of text
            if validation_function:
                cropped_img = img[y:y+h, x:x+w]
                validation_function(cropped_img, text)

    # Print extracted text with coordinates
    print("Extracted Text with Coordinates:")
    for text, coords in extracted_text_with_coords:
        print(f"Text: '{text}', Coordinates: {coords}")

    # Optionally display the image with bounding boxes
    for _, (x1, y1, x2, y2) in extracted_text_with_coords:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # cv2.imshow('Detected Text', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    output_path = "/Users/ishan/Downloads/annotated_boxes_all.jpg"
    cv2.imwrite(output_path, img)

    return extracted_text_with_coords

def process_image(image_path):
    """
    Detect outer bounding boxes in the given image, filter them by area threshold,
    and save the resulting image with detected boxes.
    
    Args:
        image_path (str): Path to the input image.
    
    Returns:
        str: Path to the saved output image.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to isolate the bounding boxes
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Identify the outer bounding box
    outer_boxes = []
    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][2] != -1:  # Check if the contour has child contours
            x, y, w, h = cv2.boundingRect(contour)
            outer_boxes.append((x, y, x + w, y + h))

    # Calculate image area
    image_height, image_width = image.shape[:2]
    image_area = image_height * image_width

    # Set the threshold dynamically as a fraction of the image area
    min_area_threshold = 0.01 * image_area  # 1% of the total image area

    # Filter bounding boxes by size
    filtered_boxes = []
    for x1, y1, x2, y2 in outer_boxes:
        area = (x2 - x1) * (y2 - y1)  # Calculate area of the bounding box
        if area > min_area_threshold:
            filtered_boxes.append((x1, y1, x2, y2))

    # Visualize the detected outer boxes
    for x1, y1, x2, y2 in filtered_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue rectangle for the outer box

    # Save the result with a modified filename
    base_name, ext = os.path.splitext(os.path.basename(image_path))
    directory = os.path.dirname(image_path)
    output_path = os.path.join(directory, f"{base_name}_detected{len(filtered_boxes)}{ext}")
    cv2.imwrite(output_path, image)

    print(f"Detected outer box saved to: {output_path}")
    return output_path

def detect_outer_boxes(image_path, min_width=100, min_height=50):
    """
    Detects outer bounding boxes in the input image, filters them by minimum width and height,
    and saves the result with visualized boxes.

    Args:
        image_path (str): Path to the input image.
        min_width (int): Minimum width of the bounding box.
        min_height (int): Minimum height of the bounding box.

    Returns:
        str: Path to the saved output image.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to isolate the bounding boxes
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Identify the outer bounding boxes
    outer_boxes = []
    for idx, contour in enumerate(contours):
        # Check if the contour has child contours (hierarchy value at index 2 is >= 0)
        if hierarchy[0][idx][2] != -1:
            # Get bounding box for the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out small boxes based on width and height
            if w > min_width and h > min_height:
                outer_boxes.append((x, y, x + w, y + h))

    print(f"Number of outer boxes detected: {len(outer_boxes)}")

    # Visualize the detected outer boxes
    for x1, y1, x2, y2 in outer_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue rectangle for the outer boxes

    # Save the result with a modified filename
    base_name, ext = os.path.splitext(os.path.basename(image_path))
    directory = os.path.dirname(image_path)
    output_path = os.path.join(directory, f"{base_name}_detected{ext}")
    cv2.imwrite(output_path, image)

    print(f"Detected outer box saved to: {output_path}")
    return output_path, outer_boxes



def boundingBoxText(outer_boxes, image):
    # Initialize the result dictionary
    text_in_boxes = {}
    
    # Loop through each bounding box
    for idx, (x1, y1, x2, y2) in enumerate(outer_boxes):
        # Crop the region of the bounding box
        cropped_region = image[y1:y2, x1:x2]
        
        # Use Tesseract to extract text from the cropped region
        extracted_text = pytesseract.image_to_string(cropped_region, lang='eng').strip()
        
        # Add the result to the dictionary
        text_in_boxes[idx] = (f"Box {idx + 1}", extracted_text)
    
    # Print the results (bounding box number and extracted text)
    for box_num, (label, text) in text_in_boxes.items():
        print(f"{label}: {text}")