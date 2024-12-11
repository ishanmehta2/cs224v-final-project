from image_processing import (
    extract_text_with_coordinates, process_image, detect_outer_boxes, boundingBoxText
)
from text_processing import clean_text

from gpt_integration import (
    refine_text_with_gpt, extract_text_from_images_with_gpt4o, refine_gpt_extracted_text
)
from additional_processing import (
    preprocess_image, detect_bounding_boxes, filter_bounding_boxes, extract_and_refine_text, process_image_for_text_extraction, save_canvas_with_bounding_boxes, refine_extracted_text
)
from evaluation import evaluate_extraction
import cv2


def runEverything(filepath):
    image = cv2.imread(filepath)
    extract_text_with_coordinates(filepath)
    output_path = process_image(filepath)
    output_path, outer_boxes = detect_outer_boxes(filepath, min_width=100, min_height=50)
    boundingBoxText(outer_boxes, image)
    results = process_image_for_text_extraction(output_path)
    save_directory = "/Users/ishan/Downloads/save_path"

    preprocessed_image, image = preprocess_image(image_path)
    filtered_boxes = filter_bounding_boxes(detect_bounding_boxes(preprocessed_image))
    
    canvas_save_path = save_canvas_with_bounding_boxes(image, filtered_boxes, save_directory)
    valid_bounding_boxes = refine_extracted_text(image, results)
    api_key = "add_api_key"
    refined_results = refine_text_with_gpt(valid_bounding_boxes, api_key)
    print(refined_results)
    gpt_extracted_results = extract_text_from_images_with_gpt4o(refined_results, image_path, api_key)
    print(gpt_extracted_results)
    refined_gpt_results = refine_gpt_extracted_text(gpt_extracted_results, api_key)
    print(refined_gpt_results)


    evaluation_results, weighted_averages = evaluate_extraction(refined_results, refined_gpt_results)
    # printing
    for box, metrics in evaluation_results.items():
        print(f"{box}:")
        print(f"  Similarity: {metrics['similarity']:.2f}")
        print(f"  Precision: {metrics['precision']:.2f}")
        print(f"  Recall: {metrics['recall']:.2f}")
        print(f"  F1 Score: {metrics['f1_score']:.2f}")
        print(f"  Refined Text: {metrics['refined_text']}")
        print(f"  GPT Extracted Text: {metrics['gpt_text']}")
        print("=" * 50)
    
    # print weights
    print("Weighted Averages:")
    print(f"  Similarity: {weighted_averages['similarity']:.2f}")
    print(f"  Precision: {weighted_averages['precision']:.2f}")
    print(f"  Recall: {weighted_averages['recall']:.2f}")
    print(f"  F1 Score: {weighted_averages['f1_score']:.2f}")
    
    return refined_results, gpt_extracted_results

def main():
    image_path = "/Users/ishan/Downloads/newspaper_img.jpeg"
    runEverything(image_path)

if __name__ == "__main__":
    main()
