from PIL import Image
import requests
from io import BytesIO
import openai
import base64
import os

from typing import List, Tuple
from difflib import SequenceMatcher
import sacrebleu
from collections import Counter

import json
import string

def clean_text(input_text: str) -> str:
    
    translator = str.maketrans('', '', string.punctuation)
    cleaned_text = input_text.translate(translator).lower()
    cleaned_text = ''.join(char for char in cleaned_text if char.isalnum() or char.isspace())
    
    return cleaned_text

client = openai.OpenAI(
  api_key='put openAI key here (we have left ours out for public viewing)',
  base_url="https://api.openai.com/v1"
)

def extract_text_from_image_with_gpt4o(image, img_type="image/png", prompt="Extract the text from this image. Output only the text from the image."):
  
    try:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        img_b64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"},
                        },
                    ],
                }
            ],
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"An error occurred: {str(e)}"

def refine_text_with_gpt4o(input_text):

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Please clean and refine the following text. Return just the refined text, nothing else. Specifically:\n"
                        "1. Correct formatting issues, such as improper line breaks, while maintaining paragraph structure.\n"
                        '2. Remove any unnecessary extra spaces or random formatting artifacts.\n'
                        '3. Fix any misspellings and misformatting of words.\n'
                        f"Input Text: {input_text}"
                    ),
                }
            ],
        )

        refined_text = response.choices[0].message.content
        return refined_text
    except Exception as e:
        return f"An error occurred: {str(e)}"
    

def calculate_precision_recall_f1(true_tokens, extracted_tokens):

    true_counter = Counter(true_tokens)
    extracted_counter = Counter(extracted_tokens)

    # calculate overlap
    overlap = sum((true_counter & extracted_counter).values())
    precision = overlap / len(extracted_tokens) if extracted_tokens else 0
    recall = overlap / len(true_tokens) if true_tokens else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1


def evaluate_extractions_weighted(pairs: List[Tuple[str, str]]) -> dict:

    total_length = 0
    weighted_similarity_sum = 0
    weighted_bleu_sum = 0
    weighted_precision_sum = 0
    weighted_recall_sum = 0
    weighted_f1_sum = 0

    results = []

    for true_text, extracted_text in pairs:
        # calculate weights based on the length of the true text
        length = len(true_text)  # weight by number of characters
        total_length += length

        similarity = SequenceMatcher(None, true_text, extracted_text).ratio()
        weighted_similarity_sum += similarity * length

        bleu_score = sacrebleu.sentence_bleu(extracted_text, [true_text]).score / 100
        weighted_bleu_sum += bleu_score * length

        true_tokens = true_text.split()
        extracted_tokens = extracted_text.split()
        precision, recall, f1 = calculate_precision_recall_f1(true_tokens, extracted_tokens)

        weighted_precision_sum += precision * length
        weighted_recall_sum += recall * length
        weighted_f1_sum += f1 * length

        # record everything
        results.append({
            "true_text": true_text,
            "extracted_text": extracted_text,
            "similarity": similarity,
            "bleu_score": bleu_score,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "length": length,
        })

    # get weighted averages
    weighted_avg_similarity = weighted_similarity_sum / total_length if total_length else 0
    weighted_avg_bleu = weighted_bleu_sum / total_length if total_length else 0
    weighted_avg_precision = weighted_precision_sum / total_length if total_length else 0
    weighted_avg_recall = weighted_recall_sum / total_length if total_length else 0
    weighted_avg_f1 = weighted_f1_sum / total_length if total_length else 0

    # summary metrics
    summary = {
        "results": results,
        "weighted_average_similarity": weighted_avg_similarity,
        "weighted_average_bleu_score": weighted_avg_bleu,
        "weighted_average_precision": weighted_avg_precision,
        "weighted_average_recall": weighted_avg_recall,
        "weighted_average_f1_score": weighted_avg_f1,
        "total_length": total_length,
    }

    return summary




def main():
    folder_path = "/Users/ohmpatel/Downloads/extraction_testing_jsons"
    data_jsons = []
    for entry in os.listdir(folder_path)[:10]:
        full_path = os.path.join(folder_path, entry)  # Create the full path
        if os.path.isfile(full_path):
            with open(full_path, 'r') as file:
                data = json.load(file)
                data_jsons.append(data)
    
    extractions = {} # map index to dictionary of ids to image, true text, extraction
    for i, paper_json in enumerate(data_jsons):
        extractions[i] = []
        print(f"Downloading image {i} ...")
        response = requests.get(paper_json['scan']['jp2_url'])
        img = Image.open(BytesIO(response.content))
        for box_info in paper_json['bboxes']:
            bbox = box_info['bbox']
            cropped_image = img.crop((bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1']))
            print("Extracting text with vision model ...")
            raw_extracted = extract_text_from_image_with_gpt4o(cropped_image)
            true_text = box_info['raw_text']
            extractions[i].append([cropped_image, true_text, raw_extracted])
    
    test_tuples = [] # true, generated
    for i in extractions.keys():
        for img, true, gen in extractions[i]:
            if true == "":
                continue
            test_tuples.append((clean_text(true), clean_text(gen)))
    evaluate_extractions_weighted(test_tuples)

    refined_test_tuples = [] # true, generated
    for i in extractions.keys():
        print(f"Refining on key {i}")
        count = 0
        for img, true, gen in extractions[i]:
            print(count)
            count += 1
            if true == "":
                print("null text, skipping ...")
                continue
            true_ref = refine_text_with_gpt4o(true)
            gen_ref = refine_text_with_gpt4o(gen)
            true_ref = clean_text(true_ref)
            gen_ref = clean_text(gen_ref)
            refined_test_tuples.append((true_ref, gen_ref))
    evaluate_extractions_weighted(refined_test_tuples)


if __name__ == "__main__":
    main()
