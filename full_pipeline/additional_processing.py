import cv2
import pytesseract
import numpy as np

def preprocess_image(image_path):

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    return thresh, image

def detect_bounding_boxes(preprocessed_image):
    
    contours, hierarchy = cv2.findContours(preprocessed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    outer_boxes = []
    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][2] != -1:  # check for child contours
            x, y, w, h = cv2.boundingRect(contour)
            outer_boxes.append((x, y, x + w, y + h))
    
    return outer_boxes

def filter_bounding_boxes(outer_boxes):
    
    box_areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in outer_boxes]
    min_area_threshold = 0.2 * np.mean(box_areas) if box_areas else 0

    filtered_boxes = [
        box for box in outer_boxes if (box[2] - box[0]) * (box[3] - box[1]) > min_area_threshold
    ]

    return filtered_boxes

def extract_and_refine_text(image, filtered_boxes):
    
    refined_results = {}

    for idx, (x1, y1, x2, y2) in enumerate(filtered_boxes, start=1):
        cropped_image = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        processed_image = cv2.fastNlMeansDenoising(binary_image, None, 30, 7, 21)

        refined_text = pytesseract.image_to_string(processed_image, lang='eng').strip()
        cleaned_text = " ".join(refined_text.split())

        refined_results[f"Box {idx}"] = {
            "coordinates": (x1, y1, x2, y2),
            "text": cleaned_text
        }

    return refined_results

def process_image_for_text_extraction(image_path):
    
    preprocessed_image, original_image = preprocess_image(image_path)

    outer_boxes = detect_bounding_boxes(preprocessed_image)
    filtered_boxes = filter_bounding_boxes(outer_boxes)
    final_output_with_coordinates = extract_and_refine_text(original_image, filtered_boxes)

    return final_output_with_coordinates


def save_canvas_with_bounding_boxes(image, filtered_boxes, save_dir):

    canvas_height, canvas_width = image.shape[:2]

    blank_canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    for x1, y1, x2, y2 in filtered_boxes:
        cv2.rectangle(blank_canvas, (x1, y1), (x2, y2), (255, 0, 0), 3) 

    base_name = "filtered_bounding_boxes"
    counter = 1
    output_path_filtered = os.path.join(save_dir, f"{base_name}_{counter}.jpg")
    while os.path.exists(output_path_filtered):
        counter += 1
        output_path_filtered = os.path.join(save_dir, f"{base_name}_{counter}.jpg")

    cv2.imwrite(output_path_filtered, blank_canvas)

    print(f"Clean canvas with filtered boxes saved to: {output_path_filtered}")
    return output_path_filtered


def refine_extracted_text(image, final_output_with_coordinates):
    
    valid_bounding_boxes = {}

    for idx, (label, data) in enumerate(final_output_with_coordinates.items()):
        coordinates = data['coordinates']
        x1, y1, x2, y2 = coordinates
        extracted_text = data['text']

        if not extracted_text.strip():
            continue

        cropped_image = image[y1:y2, x1:x2]

        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        processed_image = cv2.fastNlMeansDenoising(binary_image, None, 30, 7, 21)

        refined_text = pytesseract.image_to_string(processed_image, lang='eng').strip()
        cleaned_text = " ".join(refined_text.split())

        valid_bounding_boxes[idx] = {
            "coordinates": coordinates,
            "refined_text": cleaned_text
        }

    return valid_bounding_boxes
