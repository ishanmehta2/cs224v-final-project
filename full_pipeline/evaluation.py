

def evaluate_extraction(refined_results, gpt_extracted_results):
    
    results = {}
    weighted_totals = {
        "similarity": 0,
        "precision": 0,
        "recall": 0,
        "f1_score": 0,
        "total_weight": 0  # cumulative weight for normalization
    }

    for num_key, refined_data in refined_results.items():
        gpt_key = f"Box {num_key}"
        
        if gpt_key not in gpt_extracted_results:
            print(f"Key mismatch: {gpt_key} not found in gpt_extracted_results")
            continue

        refined_text = refined_data.get("refined_text", "").strip()
        gpt_text = gpt_extracted_results[gpt_key].get("refined_text", "").strip()

        similarity = SequenceMatcher(None, refined_text, gpt_text).ratio()

        refined_tokens = refined_text.split()
        gpt_tokens = gpt_text.split()

        common_tokens = set(refined_tokens).union(gpt_tokens)
        refined_binary = np.array([1 if token in refined_tokens else 0 for token in common_tokens])
        gpt_binary = np.array([1 if token in gpt_tokens else 0 for token in common_tokens])

        precision = precision_score(refined_binary, gpt_binary, zero_division=0)
        recall = recall_score(refined_binary, gpt_binary, zero_division=0)
        f1 = f1_score(refined_binary, gpt_binary, zero_division=0)

        if max(similarity, f1) < 0.4: # implies text is extracted differently
            continue

        # weights
        weight = len(refined_tokens)
        weighted_totals["similarity"] += similarity * weight
        weighted_totals["precision"] += precision * weight
        weighted_totals["recall"] += recall * weight
        weighted_totals["f1_score"] += f1 * weight
        weighted_totals["total_weight"] += weight

        # store individual results
        results[gpt_key] = {
            "similarity": similarity,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "refined_text": refined_text,
            "gpt_text": gpt_text,
        }

    # calculate weighted averages
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
