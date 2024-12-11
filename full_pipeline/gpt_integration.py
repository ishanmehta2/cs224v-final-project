import openai
import base64
from io import BytesIO
from PIL import Image

def refine_text_with_gpt(valid_bounding_boxes, api_key, model="gpt-4o"):
    openai.api_key = api_key

    refined_results = {}

    for idx, data in valid_bounding_boxes.items():
        txt = data['refined_text']
        coordinates = data['coordinates']

        # skip empty text
        if not txt.strip():
            continue

        estimated_tokens = int(len(txt.split()) * 1.5)  # adjust multiplier as needed
        max_tokens = min(4096, max(100, estimated_tokens))  # gpt has a limit

        # call gpt
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

            cleaned_text_final = response['choices'][0]['message']['content']

            cleaned_paragraph = " ".join(cleaned_text_final.split())

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
                                
    openai.api_key = api_key

    original_image = cv2.imread(original_image_path)
    if original_image is None:
        raise FileNotFoundError(f"Original image not found at {original_image_path}")

    gpt_results = {}

    for idx, data in refined_output.items():
        coordinates = data["coordinates"]
        refined_text = data["refined_text"]

        x1, y1, x2, y2 = coordinates
        cropped_cv2_image = original_image[y1:y2, x1:x2]

        cropped_pil_image = Image.fromarray(cv2.cvtColor(cropped_cv2_image, cv2.COLOR_BGR2RGB))

        buffer = BytesIO()
        cropped_pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        img_b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        prompt = prompt_template.replace("{text}", refined_text)

        try:
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

            gpt_extracted_text = response["choices"][0]["message"]["content"]

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
    # this is similar to the first function in the file, can use that but I had this written
    
    openai.api_key = api_key

    refined_results = {}

    for key, data in gpt_extracted_results.items():
        txt = data.get("extracted_text", "").strip()
        coordinates = data.get("coordinates", ())

        if not txt:
            continue

        estimated_tokens = int(len(txt.split()) * 1.5)  # adjust multiplier as needed
        max_tokens = min(4096, max(100, estimated_tokens))  # gpt has a limit

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

            cleaned_text_final = response['choices'][0]['message']['content']

            cleaned_paragraph = " ".join(cleaned_text_final.split())

            refined_results[key] = {
                "coordinates": coordinates,
                "refined_text": cleaned_paragraph,
            }

        except Exception as e:
            print(f"Error processing key {key}: {e}")
            continue

    print(f"Refined {len(refined_results.keys())} entries from GPT-extracted results.")
    return refined_results
