import openai
import base64
from io import BytesIO
from PIL import Image

def refine_text_with_gpt(valid_bounding_boxes, api_key, model="gpt-4o"):
    """
    Refines text from bounding boxes using GPT-4.

    Args:
        valid_bounding_boxes (dict): Dictionary of bounding boxes with coordinates and refined text.
        api_key (str): OpenAI API key for authentication.
        model (str): GPT model to use (default: "gpt-4").

    Returns:
        dict: Dictionary with bounding box indices, coordinates, and GPT-refined text.
    """
    # Set the OpenAI API key
    openai.api_key = api_key

    refined_results = {}

    for idx, data in valid_bounding_boxes.items():
        # Extract refined text and coordinates
        txt = data['refined_text']
        coordinates = data['coordinates']

        # Skip empty refined text
        if not txt.strip():
            continue

        # Dynamically calculate max_tokens
        estimated_tokens = int(len(txt.split()) * 1.5)  # Adjust multiplier as needed
        max_tokens = min(4096, max(100, estimated_tokens))  # Cap at GPT-4 limit, ensure a minimum of 100 tokens

        # Call GPT-4 to refine the text
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert editor."},
                    {
                        "role": "user",
                        "content": (
                            "You are an expert editor. Please rewrite the following text in proper English, "
                            "preserving all details, nuances, and structure. Do not summarize or make the text "
                            "more succinct—simply ensure it is grammatically correct, readable, and properly formatted. "
                            f"Here is the text:\n\n{txt}"
                        ),
                    },
                ],
                max_tokens=max_tokens,
            )

            # Extract the rewritten text
            cleaned_text_final = response['choices'][0]['message']['content']

            # Clean and format the final text into a single paragraph
            cleaned_paragraph = " ".join(cleaned_text_final.split())

            # Add to refined results
            refined_results[idx] = {
                "coordinates": coordinates,
                "refined_text": cleaned_paragraph,
            }

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            continue
    print(len(refined_results.keys()))
    return refined_results


def extract_text_from_images_with_gpt4o(refined_output, original_image_path, api_key, img_type="image/png", prompt_template="You are an expert editor. Please rewrite the following text in proper English, "
                            "preserving all details, nuances, and structure. Do not summarize or make the text "
                            "more succinct—simply ensure it is grammatically correct, readable, and properly formatted."):
    """
    Processes cropped images from bounding box coordinates and extracts text using GPT-4o.

    Args:
        refined_output (dict): Dictionary of bounding box coordinates and refined text from previous function.
        original_image_path (str): Path to the original image.
        api_key (str): OpenAI API key for authentication.
        img_type (str): MIME type of the image (default is "image/png").
        prompt_template (str): Template for GPT-4o prompt.

    Returns:
        dict: Dictionary of bounding boxes with extracted text from GPT-4o.
    """
    # Set OpenAI API key
    openai.api_key = api_key

    # Load the original image
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        raise FileNotFoundError(f"Original image not found at {original_image_path}")

    gpt_results = {}

    for idx, data in refined_output.items():
        coordinates = data["coordinates"]
        refined_text = data["refined_text"]

        # Crop the region using bounding box coordinates
        x1, y1, x2, y2 = coordinates
        cropped_cv2_image = original_image[y1:y2, x1:x2]

        # Convert the cropped OpenCV image to a PIL image
        cropped_pil_image = Image.fromarray(cv2.cvtColor(cropped_cv2_image, cv2.COLOR_BGR2RGB))

        # Convert the image to Base64 string for GPT-4o
        buffer = BytesIO()
        cropped_pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        img_b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        # Create the GPT-4o prompt
        prompt = prompt_template.replace("{text}", refined_text)

        try:
            # Send the image and prompt to GPT-4o API
            response = openai.ChatCompletion.create(
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

            # Extract the text from the response
            gpt_extracted_text = response["choices"][0]["message"]["content"]

            # Store the result
            gpt_results[f"Box {idx}"] = {
                "coordinates": coordinates,
                "extracted_text": gpt_extracted_text,
            }

        except Exception as e:
            print(f"Error processing Box {idx}: {e}")
            gpt_results[f"Box {idx}"] = {
                "coordinates": coordinates,
                "extracted_text": f"Error: {str(e)}",
            }
    print(len(gpt_results.keys()))
    return gpt_results


def refine_gpt_extracted_text(gpt_extracted_results, api_key, model="gpt-4o"):
    """
    Refines text from GPT-extracted results using GPT-4.

    Args:
        gpt_extracted_results (dict): Dictionary of bounding boxes with coordinates and GPT-extracted text.
        api_key (str): OpenAI API key for authentication.
        model (str): GPT model to use (default: "gpt-4").

    Returns:
        dict: Dictionary with bounding box indices, coordinates, and GPT-refined text.
    """
    # Set the OpenAI API key
    openai.api_key = api_key

    refined_results = {}

    for key, data in gpt_extracted_results.items():
        # Extract GPT-extracted text and coordinates
        txt = data.get("extracted_text", "").strip()
        coordinates = data.get("coordinates", ())

        # Skip empty text
        if not txt:
            continue

        # Dynamically calculate max_tokens
        estimated_tokens = int(len(txt.split()) * 1.5)  # Adjust multiplier as needed
        max_tokens = min(4096, max(100, estimated_tokens))  # Cap at GPT-4 limit, ensure a minimum of 100 tokens

        # Call GPT-4 to refine the text
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert editor."},
                    {
                        "role": "user",
                        "content": (
                            "You are an expert editor. Please rewrite the following text in proper English, "
                            "preserving all details, nuances, and structure. Do not summarize or make the text "
                            "more succinct—simply ensure it is grammatically correct, readable, and properly formatted. "
                            f"Here is the text:\n\n{txt}"
                        ),
                    },
                ],
                max_tokens=max_tokens,
            )

            # Extract the rewritten text
            cleaned_text_final = response['choices'][0]['message']['content']

            # Clean and format the final text into a single paragraph
            cleaned_paragraph = " ".join(cleaned_text_final.split())

            # Add to refined results
            refined_results[key] = {
                "coordinates": coordinates,
                "refined_text": cleaned_paragraph,
            }

        except Exception as e:
            print(f"Error processing key {key}: {e}")
            continue

    print(f"Refined {len(refined_results.keys())} entries from GPT-extracted results.")
    return refined_results
