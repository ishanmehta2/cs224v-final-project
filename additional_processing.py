import cv2
import pytesseract
import numpy as np

def preprocess_image(image_path):
    """
    Loads and preprocesses the image for contour detection.

    Args:
        image_path (str): Path to the input image.

    Returns:
        numpy.ndarray: Preprocessed image.
        numpy.ndarray: Original image for reference.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    return thresh, image

def detect_bounding_boxes(preprocessed_image):
    """
    Detects bounding boxes from the preprocessed image.

    Args:
        preprocessed_image (numpy.ndarray): Thresholded image for contour detection.

    Returns:
        list: List of bounding boxes as (x1, y1, x2, y2).
    """
    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    outer_boxes = []
    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][2] != -1:  # Check for child contours
            x, y, w, h = cv2.boundingRect(contour)
            outer_boxes.append((x, y, x + w, y + h))
    
    return outer_boxes

def filter_bounding_boxes(outer_boxes):
    """
    Filters bounding boxes by dynamic area threshold.

    Args:
        outer_boxes (list): List of bounding boxes as (x1, y1, x2, y2).

    Returns:
        list: Filtered list of bounding boxes.
    """
    # Compute dynamic threshold based on mean area
    box_areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in outer_boxes]
    min_area_threshold = 0.2 * np.mean(box_areas) if box_areas else 0

    # Filter boxes by area
    filtered_boxes = [
        box for box in outer_boxes if (box[2] - box[0]) * (box[3] - box[1]) > min_area_threshold
    ]

    return filtered_boxes

def extract_and_refine_text(image, filtered_boxes):
    """
    Extracts and refines text from bounding boxes using Tesseract.

    Args:
        image (numpy.ndarray): Original image.
        filtered_boxes (list): List of filtered bounding boxes.

    Returns:
        dict: Dictionary of refined text and bounding box coordinates.
    """
    refined_results = {}

    for idx, (x1, y1, x2, y2) in enumerate(filtered_boxes, start=1):
        # Crop the region based on the coordinates
        cropped_image = image[y1:y2, x1:x2]

        # Preprocess the cropped image
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        processed_image = cv2.fastNlMeansDenoising(binary_image, None, 30, 7, 21)

        # Extract text using Tesseract
        refined_text = pytesseract.image_to_string(processed_image, lang='eng').strip()
        cleaned_text = " ".join(refined_text.split())

        # Add to results dictionary
        refined_results[f"Box {idx}"] = {
            "coordinates": (x1, y1, x2, y2),
            "text": cleaned_text
        }

    return refined_results

def process_image_for_text_extraction(image_path):
    """
    Full pipeline for detecting bounding boxes and extracting text.

    Args:
        image_path (str): Path to the input image.

    Returns:
        dict: Final dictionary with bounding box labels, coordinates, and refined text.
    """
    # Preprocess the image
    preprocessed_image, original_image = preprocess_image(image_path)

    # Detect bounding boxes
    outer_boxes = detect_bounding_boxes(preprocessed_image)

    # Filter bounding boxes
    filtered_boxes = filter_bounding_boxes(outer_boxes)

    # Extract and refine text
    final_output_with_coordinates = extract_and_refine_text(original_image, filtered_boxes)

    return final_output_with_coordinates


def save_canvas_with_bounding_boxes(image, filtered_boxes, save_dir):
    """
    Creates and saves a blank canvas with filtered bounding boxes drawn on it.

    Args:
        image (numpy.ndarray): The input image for reference dimensions.
        filtered_boxes (list): List of bounding boxes as (x1, y1, x2, y2).
        save_dir (str): Directory where the resulting canvas will be saved.

    Returns:
        str: Path to the saved canvas image.
    """
    # Get the dimensions of the input image
    canvas_height, canvas_width = image.shape[:2]

    # Create a blank white canvas
    blank_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # Draw the filtered bounding boxes on the canvas
    for x1, y1, x2, y2 in filtered_boxes:
        cv2.rectangle(blank_canvas, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue rectangles

    # Generate a unique save path based on the save directory
    base_name = "filtered_bounding_boxes"
    counter = 1
    output_path_filtered = os.path.join(save_dir, f"{base_name}_{counter}.jpg")
    while os.path.exists(output_path_filtered):
        counter += 1
        output_path_filtered = os.path.join(save_dir, f"{base_name}_{counter}.jpg")

    # Save the blank canvas with bounding boxes
    cv2.imwrite(output_path_filtered, blank_canvas)

    print(f"Clean canvas with filtered boxes saved to: {output_path_filtered}")
    return output_path_filtered


def refine_extracted_text(image, final_output_with_coordinates):
    """
    Refines text extraction for bounding boxes by preprocessing the image
    and using Tesseract for text recognition.

    Args:
        image (numpy.ndarray): The input image.
        final_output_with_coordinates (dict): Dictionary containing bounding box
                                              coordinates and initially extracted text.

    Returns:
        dict: Dictionary of valid bounding boxes with coordinates and refined text.
    """
    valid_bounding_boxes = {}

    for idx, (label, data) in enumerate(final_output_with_coordinates.items()):
        coordinates = data['coordinates']
        x1, y1, x2, y2 = coordinates
        extracted_text = data['text']

        # Skip boxes with no extracted text
        if not extracted_text.strip():
            continue

        # Crop the region based on the coordinates
        cropped_image = image[y1:y2, x1:x2]

        # Preprocess the cropped image
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        processed_image = cv2.fastNlMeansDenoising(binary_image, None, 30, 7, 21)

        # Refine the text using Tesseract
        refined_text = pytesseract.image_to_string(processed_image, lang='eng').strip()
        cleaned_text = " ".join(refined_text.split())

        # Store the refined text and coordinates with the index key
        valid_bounding_boxes[idx] = {
            "coordinates": coordinates,
            "refined_text": cleaned_text
        }

    return valid_bounding_boxes