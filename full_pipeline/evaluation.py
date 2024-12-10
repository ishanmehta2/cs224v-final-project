

def evaluate_extraction(refined_results, gpt_extracted_results):
    """
    Compares the refined text from two dictionaries using similarity, F1-score, precision, and recall.

    Args:
        refined_results (dict): Dictionary of refined results with numeric keys.
        gpt_extracted_results (dict): Dictionary of GPT-extracted results with 'Box <index>' keys.

    Returns:
        dict: Dictionary containing similarity, F1-score, precision, and recall for each box.
              Also includes weighted average scores for each metric.
    """
    results = {}
    weighted_totals = {
        "similarity": 0,
        "precision": 0,
        "recall": 0,
        "f1_score": 0,
        "total_weight": 0  # Cumulative weight for normalization
    }

    for num_key, refined_data in refined_results.items():
        # Match the corresponding key in gpt_extracted_results
        gpt_key = f"Box {num_key}"
        
        if gpt_key not in gpt_extracted_results:
            print(f"Key mismatch: {gpt_key} not found in gpt_extracted_results")
            continue

        # Retrieve texts
        refined_text = refined_data.get("refined_text", "").strip()
        gpt_text = gpt_extracted_results[gpt_key].get("refined_text", "").strip()

        # Calculate similarity
        similarity = SequenceMatcher(None, refined_text, gpt_text).ratio()

        # Tokenize texts for precision, recall, and F1-score
        refined_tokens = refined_text.split()
        gpt_tokens = gpt_text.split()

        # Create binary arrays for matching tokens
        common_tokens = set(refined_tokens).union(gpt_tokens)
        refined_binary = np.array([1 if token in refined_tokens else 0 for token in common_tokens])
        gpt_binary = np.array([1 if token in gpt_tokens else 0 for token in common_tokens])

        # Calculate precision, recall, and F1-score
        precision = precision_score(refined_binary, gpt_binary, zero_division=0)
        recall = recall_score(refined_binary, gpt_binary, zero_division=0)
        f1 = f1_score(refined_binary, gpt_binary, zero_division=0)

        # # If similarity or F1-score is below 0.2, skip this box
        if max(similarity, f1) < 0.4: # implies text is extracted differently
            continue

        # Weight for this box (number of tokens in refined_text)
        weight = len(refined_tokens)
        weighted_totals["similarity"] += similarity * weight
        weighted_totals["precision"] += precision * weight
        weighted_totals["recall"] += recall * weight
        weighted_totals["f1_score"] += f1 * weight
        weighted_totals["total_weight"] += weight

        # Store individual results
        results[gpt_key] = {
            "similarity": similarity,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "refined_text": refined_text,
            "gpt_text": gpt_text,
        }

    # Calculate weighted averages
    total_weight = weighted_totals["total_weight"]
    if total_weight > 0:
        weighted_averages = {
            "similarity": weighted_totals["similarity"] / total_weight,
            "precision": weighted_totals["precision"] / total_weight,
            "recall": weighted_totals["recall"] / total_weight,
            "f1_score": weighted_totals["f1_score"] / total_weight,
        }
    else:
        weighted_averages = {"similarity": 0, "precision": 0, "recall": 0, "f1_score": 0}

    return results, weighted_averages
