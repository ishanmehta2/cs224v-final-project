import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import os
from PIL import Image
from io import BytesIO

def extract_text_with_coordinates(image_path, validation_function=None):
    img = cv2.imread(image_path)

    # use tesseract
    d = pytesseract.image_to_data(img, output_type=Output.DICT)

    extracted_text_with_coords = []

    n_boxes = len(d['level'])
    for i in range(n_boxes):
        text = d['text'][i].strip()
        if text:  # only include non-empty text
            x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
            bounding_box = (x, y, x + w, y + h)
            extracted_text_with_coords.append((text, bounding_box))

            if validation_function:
                cropped_img = img[y:y+h, x:x+w]
                validation_function(cropped_img, text)

    print("Extracted Text with Coordinates:")
    for text, coords in extracted_text_with_coords:
        print(f"Text: '{text}', Coordinates: {coords}")

    for _, (x1, y1, x2, y2) in extracted_text_with_coords:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    output_path = "/Users/ishan/Downloads/annotated_boxes_all.jpg"
    cv2.imwrite(output_path, img)

    return extracted_text_with_coords

def process_image(image_path):

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    outer_boxes = []
    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][2] != -1:  # check if the contour has child contours
            x, y, w, h = cv2.boundingRect(contour)
            outer_boxes.append((x, y, x + w, y + h))

    # get area
    image_height, image_width = image.shape[:2]
    image_area = image_height * image_width

    min_area_threshold = 0.01 * image_area  # 1% of the total image area but this is artbitrary and can be changed

    filtered_boxes = []
    for x1, y1, x2, y2 in outer_boxes:
        area = (x2 - x1) * (y2 - y1)  
        if area > min_area_threshold:
            filtered_boxes.append((x1, y1, x2, y2))

    # visualization
    for x1, y1, x2, y2 in filtered_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3) 

    # saving
    base_name, ext = os.path.splitext(os.path.basename(image_path))
    directory = os.path.dirname(image_path)
    output_path = os.path.join(directory, f"{base_name}_detected{len(filtered_boxes)}{ext}")
    cv2.imwrite(output_path, image)

    print(f"Detected outer box saved to: {output_path}")
    return output_path

def detect_outer_boxes(image_path, min_width=100, min_height=50):
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    outer_boxes = []
    for idx, contour in enumerate(contours):
        if hierarchy[0][idx][2] != -1:
            x, y, w, h = cv2.boundingRect(contour)
            
            # filter out anything that is too small
            if w > min_width and h > min_height:
                outer_boxes.append((x, y, x + w, y + h))

    print(f"Number of outer boxes detected: {len(outer_boxes)}")

    for x1, y1, x2, y2 in outer_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)  

    # saving
    base_name, ext = os.path.splitext(os.path.basename(image_path))
    directory = os.path.dirname(image_path)
    output_path = os.path.join(directory, f"{base_name}_detected{ext}")
    cv2.imwrite(output_path, image)

    print(f"Detected outer box saved to: {output_path}")
    return output_path, outer_boxes



def boundingBoxText(outer_boxes, image):
   
    text_in_boxes = {}

    for idx, (x1, y1, x2, y2) in enumerate(outer_boxes):
        cropped_region = image[y1:y2, x1:x2]

        extracted_text = pytesseract.image_to_string(cropped_region, lang='eng').strip()

        text_in_boxes[idx] = (f"Box {idx + 1}", extracted_text)
    
    for box_num, (label, text) in text_in_boxes.items():
        print(f"{label}: {text}")
