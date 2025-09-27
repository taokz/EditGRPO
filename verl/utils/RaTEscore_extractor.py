import copy
import re
import random
import requests
import torch
from typing import Dict, List, Tuple, Any, Optional

def clean_eos_tokens(text: str) -> str:
    """Remove all variations of EOS tokens from text."""
    if not text:
        return text
    
    # Common EOS token variations - order matters!
    # First handle tokens with surrounding spaces/newlines
    eos_patterns_with_space = [
        " <|endoftext|> ", " <|end_of_text|> ", " </s> ", " <eos> ",
        " <|eot_id|> ", " [EOS] ", " [END] ", " <|im_end|> ",
        "\n<|endoftext|>\n", "\n<|end_of_text|>\n", "\n</s>\n", "\n<eos>\n",
        "\n<|endoftext|> ", " <|endoftext|>\n", "\n</s> ", " </s>\n",
    ]
    
    # Then handle tokens without spaces (replace with a space to maintain separation)
    eos_patterns_no_space = [
        "<|endoftext|>", "<|end_of_text|>", "</s>", "<eos>",
        "<|eot_id|>", "[EOS]", "[END]", "<|im_end|>",
    ]
    
    cleaned_text = text
    
    # First pass: replace patterns that already have spaces
    for pattern in eos_patterns_with_space:
        cleaned_text = cleaned_text.replace(pattern, " ")
    
    # Second pass: replace bare tokens with a space to maintain word separation
    for pattern in eos_patterns_no_space:
        cleaned_text = cleaned_text.replace(pattern, " ")
    
    # Clean up any resulting multiple spaces
    cleaned_text = ' '.join(cleaned_text.split())
    
    return cleaned_text.strip()

def get_entity_containing_sentences(text, entity_tokens=None):
    """
    Find the sentence that contains the entity tokens.
    If entity_tokens is None or empty, returns all sentences.
    """
    sentences = []
    if not text:
        return [], None

    # More robust sentence splitting
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    
    for s in raw_sentences:
        if s.strip():
            sentences.append(s.strip())

    if not sentences and text.strip(): # Handle text with no standard sentence terminators
        sentences.append(text.strip())

    if not entity_tokens:
        return sentences, None

    entity_sentence_found = None
    normalized_entity_tokens = entity_tokens.lower()
    for sentence in sentences:
        if normalized_entity_tokens in sentence.lower():
            entity_sentence_found = sentence
            break
            
    return sentences, entity_sentence_found

def replace_sentence_in_text(original_text, old_sentence, new_sentence):
    """Replaces the first occurrence of old_sentence with new_sentence in original_text."""
    if not old_sentence or not new_sentence or old_sentence == new_sentence:
        return original_text
    
    sentences, _ = get_entity_containing_sentences(original_text)
    modified_sentences = []
    replaced = False
    for s in sentences:
        if not replaced and s == old_sentence:
            modified_sentences.append(new_sentence)
            replaced = True
        else:
            modified_sentences.append(s)
    
    return " ".join(modified_sentences).strip() if replaced else original_text

def replace_contradicting_sentence_rs(paragraph_text_to_modify, contradiction_detail):
    """Replace a contradicting sentence in the generated text with the reference sentence."""
    gen_entity_sentence = contradiction_detail['gen_entity']['sentence']
    ref_entity_sentence = clean_eos_tokens(contradiction_detail['ref_entity']['sentence'])  # Clean EOS tokens

    if not gen_entity_sentence or not ref_entity_sentence or gen_entity_sentence == ref_entity_sentence:
        return paragraph_text_to_modify, None

    new_paragraph_text = replace_sentence_in_text(paragraph_text_to_modify, gen_entity_sentence, ref_entity_sentence)
    if new_paragraph_text == paragraph_text_to_modify: 
        return paragraph_text_to_modify, None

    return new_paragraph_text, {
        'gen_token': contradiction_detail['gen_entity']['text_span'],
        'ref_token': contradiction_detail['ref_entity']['text_span'],
        'old_sentence': gen_entity_sentence, 
        'new_sentence': ref_entity_sentence,
        'gen_type': contradiction_detail['gen_entity']['type'],
        'ref_type': contradiction_detail['ref_entity']['type']
    }

def delete_fp_sentence_rs(paragraph_text_to_modify, fp_entity_detail):
    """Delete a sentence containing a false positive entity."""
    fp_sentence_to_remove = fp_entity_detail['sentence']
    if not paragraph_text_to_modify.strip() or not fp_sentence_to_remove:
        return paragraph_text_to_modify, None

    all_sentences_in_paragraph, _ = get_entity_containing_sentences(paragraph_text_to_modify)
    remaining_sentences = [s for s in all_sentences_in_paragraph if s != fp_sentence_to_remove]
    
    if len(remaining_sentences) == len(all_sentences_in_paragraph): # No sentence removed
        return paragraph_text_to_modify, None

    new_paragraph_text = " ".join(remaining_sentences).strip()
    return new_paragraph_text, {'deleted_sentence': fp_sentence_to_remove, 'fp_token': fp_entity_detail['text_span']}

def add_fn_sentence_rs(paragraph_text_to_modify, fn_entity_detail):
    """Add a sentence containing a false negative entity to the text."""
    fn_sentence_to_add = clean_eos_tokens(fn_entity_detail['sentence'])  # Clean EOS tokens before adding
    if not fn_sentence_to_add: 
        return paragraph_text_to_modify, None
        
    current_sentences, _ = get_entity_containing_sentences(paragraph_text_to_modify)
    if fn_sentence_to_add in current_sentences: 
        return paragraph_text_to_modify, None

    # Clean the paragraph text as well before concatenation
    clean_paragraph = clean_eos_tokens(paragraph_text_to_modify)
    new_paragraph_text = (clean_paragraph.strip() + " " + fn_sentence_to_add).strip()
    return new_paragraph_text, {'added_sentence': fn_sentence_to_add, 'fn_token': fn_entity_detail['text_span']}

def cosine_similarity(tensor1, tensor2):
    """Calculate cosine similarity between two tensors."""
    if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
        return 0.0
    
    # Handle empty tensors
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return 0.0
        
    # Normalize tensors
    tensor1_norm = torch.nn.functional.normalize(tensor1, p=2, dim=-1)
    tensor2_norm = torch.nn.functional.normalize(tensor2, p=2, dim=-1)
    
    # Calculate cosine similarity
    if tensor1_norm.ndim == 1 and tensor2_norm.ndim == 1:
        return torch.dot(tensor1_norm, tensor2_norm)
    else:
        # Handle other dimensions if needed
        return 0.0

def replace_fp_sentence_complex_rs(paragraph_text_to_modify, fp_entity_info_trigger, ref_entities_info_grouped_by_sentence, similarity_threshold):
    """
    Replace a false positive sentence with the reference sentence that maximizes 
    the sum of maximum cosine similarities between entities, as per equation (2).
    """
    fp_sentence_text = fp_entity_info_trigger['sentence']
    if not fp_sentence_text or not ref_entities_info_grouped_by_sentence:
        return paragraph_text_to_modify, None
    
    # Get all entities in the false positive sentence
    fp_entities = []
    for fp_entity in fp_entity_info_trigger.get('sentence_entities', [fp_entity_info_trigger]):
        if 'embedding' in fp_entity:
            fp_entities.append(fp_entity)
    
    # If no entities with embeddings, fall back to random selection
    if not fp_entities:
        if ref_entities_info_grouped_by_sentence:
            candidate_sentences = list(ref_entities_info_grouped_by_sentence.keys())
            if candidate_sentences:
                best_ref_sentence_text = random.choice(candidate_sentences)
                if best_ref_sentence_text != fp_sentence_text:
                    new_paragraph_text = replace_sentence_in_text(paragraph_text_to_modify, fp_sentence_text, best_ref_sentence_text)
                    return new_paragraph_text, {
                        'fp_trigger_token': fp_entity_info_trigger['text_span'],
                        'old_sentence': fp_sentence_text, 
                        'new_sentence': best_ref_sentence_text,
                        'selection': 'random (no embeddings)'
                    }
        return paragraph_text_to_modify, None
    
    # Calculate equation (2) for each reference sentence
    best_score = -float('inf')
    best_ref_sentence = None
    
    for ref_sentence, ref_entities_list in ref_entities_info_grouped_by_sentence.items():
        # Skip same sentence
        if ref_sentence == fp_sentence_text:
            continue
            
        # Get embeddings from reference entities
        ref_entities_with_embeddings = [entity for entity in ref_entities_list if 'embedding' in entity]
        
        # Calculate sum(max(cos(er, e))) for all er in fp_entities, e in ref_entities
        if ref_entities_with_embeddings:
            score = 0
            for fp_entity in fp_entities:
                # For each fp entity, find max similarity with any reference entity
                max_sim = 0
                fp_embedding = fp_entity['embedding']
                if isinstance(fp_embedding, list):  # Convert to tensor if needed
                    fp_embedding = torch.tensor(fp_embedding)
                
                for ref_entity in ref_entities_with_embeddings:
                    ref_embedding = ref_entity['embedding']
                    if isinstance(ref_embedding, list):  # Convert to tensor if needed
                        ref_embedding = torch.tensor(ref_embedding)
                    
                    # Calculate cosine similarity
                    try:
                        sim = cosine_similarity(fp_embedding, ref_embedding).item()
                        max_sim = max(max_sim, sim)
                    except:
                        # Handle potential errors in similarity calculation
                        pass
                
                score += max_sim
            
            # Normalize the score by the number of entities
            if fp_entities:
                score /= len(fp_entities)
            
            # Update best sentence if score is higher
            if score > best_score:
                best_score = score
                best_ref_sentence = ref_sentence
    
    # Only replace if the best score exceeds the threshold
    if best_ref_sentence and best_score >= similarity_threshold * len(fp_entities):
        clean_best_ref_sentence = clean_eos_tokens(best_ref_sentence)  # Clean EOS tokens
        new_paragraph_text = replace_sentence_in_text(paragraph_text_to_modify, fp_sentence_text, clean_best_ref_sentence)
        if new_paragraph_text != paragraph_text_to_modify:
            return new_paragraph_text, {
                'fp_trigger_token': fp_entity_info_trigger['text_span'],
                'old_sentence': fp_sentence_text, 
                'new_sentence': clean_best_ref_sentence,
                'similarity_score': best_score,
                'selection': 'maximum similarity'
            }
    
    return paragraph_text_to_modify, None

# def process_text_with_server_analysis(gen_text, analysis_result, num_operations=1, operation_priority=None, similarity_threshold=0.6):
def process_text_with_server_analysis(gen_text, analysis_result, num_operations=1, operation_priority=None, similarity_threshold=0):
    """
    Process a generated text using analysis results from the RaTEScore server.
    
    Args:
        gen_text: The generated text to modify
        analysis_result: Analysis result from the server for this text
        num_operations: Maximum number of operations to perform
        operation_priority: List of operation types in order of priority
        similarity_threshold: Cosine similarity threshold for entity matching (0.0 to 1.0)
                            - 0.0: Accept any replacement
                            - 0.6: Default, moderate similarity required
                            - 1.0: Require perfect match
        
    Returns:
        Tuple of (modified_text, change_flag, operations_log)
    """
    if operation_priority is None:
        # Updated priority: Complex FP replacement, then Delete FPs, then Add FNs
        operation_priority = ["replace_fp_complex", "delete", "add"] 
        ### operation_priority = ["replace_fp_complex", "delete"] # just for rebuttal, prior implementation

    modified_text = gen_text
    operations_log = []
    change_flag = False
    
    if not gen_text.strip():
        return gen_text, False, []

    # Clean the input text first
    gen_text = clean_eos_tokens(gen_text)
    
    # Extract data from analysis result
    contradictions = analysis_result.get("contradictions", [])
    false_positives = analysis_result.get("false_positives", [])
    false_negatives = analysis_result.get("false_negatives", [])
    ref_entities_grouped_by_sentence = analysis_result.get("ref_entities_grouped_by_sentence", {})
    
    # Clean all reference sentences
    for sentence in ref_entities_grouped_by_sentence:
        for entity in ref_entities_grouped_by_sentence[sentence]:
            if 'sentence' in entity:
                entity['sentence'] = clean_eos_tokens(entity['sentence'])
    
    processed_sentences = set()

    for op_count in range(num_operations):
        current_op_performed_this_cycle = False
        
        for op_type in operation_priority:
            if current_op_performed_this_cycle: 
                break

            # Replace false positives with complex handling
            if op_type == "replace_fp_complex" and false_positives:
                for fp_trigger in false_positives:
                    original_fp_sent_text = fp_trigger['sentence']
                    if original_fp_sent_text not in processed_sentences:
                        new_text, op_info = replace_fp_sentence_complex_rs(
                            modified_text, fp_trigger, ref_entities_grouped_by_sentence, similarity_threshold
                        )
                        if op_info:
                            # Successful replacement
                            modified_text, change_flag, current_op_performed_this_cycle = new_text, True, True
                            operations_log.append({'operation': 'replace_fp_complex', 'details': op_info})
                            processed_sentences.add(original_fp_sent_text)
                            processed_sentences.add(op_info['new_sentence'])
                            break
                        else:
                            # Failed replacement - log the attempt
                            operations_log.append({
                                'operation': 'replace_fp_complex_failed', 
                                'details': {
                                    'attempted_sentence': original_fp_sent_text,
                                    'fp_token': fp_trigger.get('text_span', ''),
                                    'reason': 'No suitable replacement found (similarity threshold not met or no embeddings)'
                                }
                            })
                            # Continue trying other FP sentences, don't break here

            # Delete false positives
            elif op_type == "delete" and false_positives:
                for fp in false_positives:
                    original_fp_sent = fp['sentence']
                    if original_fp_sent not in processed_sentences:
                        new_text, op_info = delete_fp_sentence_rs(modified_text, fp)
                        if op_info:
                            modified_text, change_flag, current_op_performed_this_cycle = new_text, True, True
                            operations_log.append({'operation': 'delete_fp_simple', 'details': op_info})
                            processed_sentences.add(original_fp_sent)
                            break

            # Add false negatives
            elif op_type == "add" and false_negatives:
                # Get a list of false negative sentences that haven't been processed yet
                available_fn = [fn for fn in false_negatives if fn['sentence'] not in processed_sentences]
                
                # If there are available false negatives, randomly select one
                if available_fn:
                    # Randomly select a false negative
                    selected_fn = random.choice(available_fn)
                    
                    # Clean the sentence before using it
                    if 'sentence' in selected_fn:
                        selected_fn['sentence'] = clean_eos_tokens(selected_fn['sentence'])
                    
                    new_text, op_info = add_fn_sentence_rs(modified_text, selected_fn)
                    if op_info:
                        modified_text, change_flag, current_op_performed_this_cycle = new_text, True, True
                        operations_log.append({'operation': 'add_fn', 'details': op_info})
                        processed_sentences.add(op_info['added_sentence'])
        
        if not current_op_performed_this_cycle:
            break
    
    # Fallback: If the paragraph becomes empty after processing, use add operation
    if not modified_text.strip() and false_negatives:
        # Get a list of false negative sentences that haven't been processed yet
        available_fn = [fn for fn in false_negatives if fn['sentence'] not in processed_sentences]
        
        if available_fn:
            # Randomly select a false negative
            selected_fn = random.choice(available_fn)
            
            # Clean the sentence before using it
            if 'sentence' in selected_fn:
                selected_fn['sentence'] = clean_eos_tokens(selected_fn['sentence'])
            
            add_text, add_op_info = add_fn_sentence_rs(modified_text, selected_fn)
            if add_op_info:
                modified_text, change_flag = add_text, True
                operations_log.append({'operation': 'add_fn_empty_fallback', 'details': add_op_info})
            
    # Final cleanup of any remaining EOS tokens
    modified_text = clean_eos_tokens(modified_text)
    
    return modified_text, change_flag, operations_log

def process_texts_with_ratescore(
    gen_candidates_per_prompt: List[List[str]], 
    gt_texts: List[str], 
    num_operations: int = 1, 
    index_prompt: Optional[List[int]] = None, 
    operation_priority: Optional[List[str]] = None, 
    server_url: Optional[str] = None
) -> Tuple[List[List[str]], bool, Dict[str, Any]]:
    """
    Process lists of generated candidate texts for corresponding ground truth texts using the RaTEScore server.
    Each ground truth text can have multiple generated candidates.
    
    Args:
        gen_candidates_per_prompt: List of lists of strings. Outer list corresponds to prompts (GTs).
                                   Inner list contains generated candidate texts for that prompt.
                                   e.g., [["cand1_gt1", "cand2_gt1"], ["cand1_gt2"]]
        gt_texts: List of ground truth strings. Must match the length of the outer list of gen_candidates_per_prompt.
                  e.g., ["gt1", "gt2"]
        num_operations: Maximum number of operations to perform per candidate text.
        index_prompt: List of prompt indices (0-based referring to gt_texts) to process. 
                      If None or [-1], a random half of all prompts will be processed.
        operation_priority: List of operation types in order of priority for process_text_with_server_analysis.
        server_url: URL of the RaTEScore server (default: http://localhost:5001/analyze).
        
    Returns:
        Tuple of (modified_texts_nested, overall_change_flag, operations_info)
        modified_texts_nested: List[List[str]] with the same structure as gen_candidates_per_prompt, containing modified texts.
        overall_change_flag: Boolean, True if any text was modified.
        operations_info: Dict containing details of operations performed.
                         Structure: {"total_operations": X, 
                                     "per_prompt": {
                                         "prompt_idx_str": {
                                             "total_ops_for_prompt": Y, 
                                             "candidates_summary": [
                                                 {"candidate_original_index": Z, "count": W, "operations": [...]}, ...
                                             ]
                                         }, ...
                                     },
                                     "error": "optional_error_string"
                                    }
    """
    if not gen_candidates_per_prompt:
        return [], False, {"total_operations": 0, "per_prompt": {}}
    
    if not gt_texts or len(gt_texts) != len(gen_candidates_per_prompt):
        raise ValueError(
            "Ground truth texts must be provided and match the number of prompts in gen_candidates_per_prompt."
        )

    # Initialize results with original candidate texts (deep copy structure)
    final_modified_texts_nested = [list(cands) for cands in gen_candidates_per_prompt]
    overall_change_flag = False
    overall_operations_info: Dict[str, Any] = {"total_operations": 0, "per_prompt": {}}

    if server_url is None:
        server_url = "http://localhost:5001/analyze"

    # Determine which prompts to process
    num_total_prompts = len(gt_texts)
    prompts_to_consider_indices = list(range(num_total_prompts))

    if index_prompt is None or (isinstance(index_prompt, list) and len(index_prompt) == 1 and index_prompt[0] == -1):
        # Process a random half if None or [-1]
        random.shuffle(prompts_to_consider_indices)
        num_to_select = num_total_prompts // 2 if num_total_prompts > 0 else 0
        selected_original_prompt_indices = prompts_to_consider_indices[:num_to_select]
    else:
        # Process specified prompt indices
        selected_original_prompt_indices = [idx for idx in index_prompt if 0 <= idx < num_total_prompts]

    if not selected_original_prompt_indices:
        # No prompts selected for processing, return original texts and no changes
        return final_modified_texts_nested, False, overall_operations_info

    # Prepare flattened lists for the server call from selected prompts
    flat_gen_texts_for_server: List[str] = []
    flat_gt_texts_for_server: List[str] = []
    # reconstruction_map stores (original_prompt_idx, candidate_idx_within_prompt) for each item in flat_gen_texts_for_server
    reconstruction_map: List[Tuple[int, int]] = []

    for original_prompt_idx in selected_original_prompt_indices:
        candidates_for_this_prompt = gen_candidates_per_prompt[original_prompt_idx]
        gt_for_this_prompt = clean_eos_tokens(gt_texts[original_prompt_idx])  # Clean ground truth
        for cand_idx, candidate_text in enumerate(candidates_for_this_prompt):
            flat_gen_texts_for_server.append(clean_eos_tokens(candidate_text))  # Clean candidate text
            flat_gt_texts_for_server.append(gt_for_this_prompt)
            reconstruction_map.append((original_prompt_idx, cand_idx))
            
    if not flat_gen_texts_for_server: # No actual candidates to process
        return final_modified_texts_nested, False, overall_operations_info

    try:
        response = requests.post(
            server_url,
            json={"generations": flat_gen_texts_for_server, "ground_truth": flat_gt_texts_for_server},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            error_msg = f"Server returned error: {response.status_code}, {response.text}"
            overall_operations_info["error"] = error_msg
            return final_modified_texts_nested, False, overall_operations_info
        
        result = response.json()
        if result.get("result") != "success":
            error_msg = f"Server returned error: {result}"
            overall_operations_info["error"] = error_msg
            return final_modified_texts_nested, False, overall_operations_info
        
        analysis_results_flat = result["data"]["predictions"]["analysis_results"]
        if len(analysis_results_flat) != len(flat_gen_texts_for_server):
            error_msg = f"Server returned {len(analysis_results_flat)} results for {len(flat_gen_texts_for_server)} inputs"
            overall_operations_info["error"] = error_msg
            return final_modified_texts_nested, False, overall_operations_info
        
        # Process flat server results and update the nested structure
        for i, analysis_for_candidate in enumerate(analysis_results_flat):
            original_prompt_idx, candidate_idx_in_prompt = reconstruction_map[i]
            original_candidate_text_for_processing = flat_gen_texts_for_server[i]
            
            modified_candidate_text, text_changed_for_candidate, ops_log_for_candidate = \
                process_text_with_server_analysis(
                    original_candidate_text_for_processing, 
                    analysis_for_candidate, 
                    num_operations, 
                    operation_priority
                )
            
            final_modified_texts_nested[original_prompt_idx][candidate_idx_in_prompt] = modified_candidate_text
            
            if text_changed_for_candidate:
                overall_change_flag = True
                prompt_key_str = str(original_prompt_idx)
                
                if prompt_key_str not in overall_operations_info["per_prompt"]:
                    overall_operations_info["per_prompt"][prompt_key_str] = {
                        "total_ops_for_prompt": 0,
                        "candidates_summary": []
                    }
                
                current_prompt_summary = overall_operations_info["per_prompt"][prompt_key_str]
                num_ops_this_candidate = len(ops_log_for_candidate)
                # Count only successful operations for the main counter
                successful_ops = sum(1 for op in ops_log_for_candidate if not op['operation'].endswith('_failed'))

                current_prompt_summary["candidates_summary"].append({
                    "candidate_original_index": candidate_idx_in_prompt,
                    "count": num_ops_this_candidate,  # Total operations including failed attempts
                    "successful_count": successful_ops,  # Only successful operations
                    "operations": ops_log_for_candidate
                })
                current_prompt_summary["total_ops_for_prompt"] += successful_ops
                overall_operations_info["total_operations"] += successful_ops
            elif ops_log_for_candidate:
                # Text didn't change but we have logged operations (e.g., failed attempts)
                # Still record these for transparency
                prompt_key_str = str(original_prompt_idx)
                
                if prompt_key_str not in overall_operations_info["per_prompt"]:
                    overall_operations_info["per_prompt"][prompt_key_str] = {
                        "total_ops_for_prompt": 0,
                        "candidates_summary": []
                    }
                
                current_prompt_summary = overall_operations_info["per_prompt"][prompt_key_str]
                num_ops_this_candidate = len(ops_log_for_candidate)
                successful_ops = sum(1 for op in ops_log_for_candidate if not op['operation'].endswith('_failed'))

                current_prompt_summary["candidates_summary"].append({
                    "candidate_original_index": candidate_idx_in_prompt,
                    "count": num_ops_this_candidate,  # Total operations including failed attempts
                    "successful_count": successful_ops,  # Only successful operations
                    "operations": ops_log_for_candidate
                })
                # Don't increment counters for failed-only operations
        
        return final_modified_texts_nested, overall_change_flag, overall_operations_info
        
    except Exception as e:
        error_msg = f"Error during RaTEScore processing: {str(e)}"
        overall_operations_info["error"] = error_msg
        return final_modified_texts_nested, False, overall_operations_info

# Simple test if run directly
if __name__ == "__main__":
    # Example usage with configurable server URL
    server_url_main = "http://RATESCORE_SERVER_HOST:5001/analyze"  # Your actual server URL

    # Test case 1: Single prompt, multiple candidates
    test_gen_candidates_1 = [
        ["The patient has pneumonia. The heart is enlarged.", "No pneumonia seen. Heart size is normal."]
    ]
    test_gt_texts_1 = ["The patient does not have pneumonia. The heart is normal in size."]
    
    print("Testing RaTEScore client with example 1 (1 prompt, 2 candidates):")
    print(f"GT Prompt 0: {test_gt_texts_1[0]}")
    print(f"  Candidate 0: {test_gen_candidates_1[0][0]}")
    print(f"  Candidate 1: {test_gen_candidates_1[0][1]}")
    print(f"Using server URL: {server_url_main}")
    print(f"New operation priority: ['replace_fp_complex', 'delete', 'add']")
    
    try:
        modified_texts_nested_1, changed_1, ops_info_1 = process_texts_with_ratescore(
            test_gen_candidates_1, test_gt_texts_1, 
            num_operations=3, # Apply up to 3 ops per candidate to test new priority
            index_prompt=None, # Process all (random half, here it will be the one prompt)
            server_url=server_url_main
        )
        
        print("\nResults for Example 1:")
        print(f"Overall text modified: {changed_1}")
        print(f"Total operations: {ops_info_1.get('total_operations', 0)}")

        for prompt_idx_str, prompt_details in ops_info_1.get("per_prompt", {}).items():
            print(f"\nPrompt {prompt_idx_str} (Total ops: {prompt_details['total_ops_for_prompt']}):")
            for cand_summary in prompt_details["candidates_summary"]:
                original_text = test_gen_candidates_1[int(prompt_idx_str)][cand_summary['candidate_original_index']]
                modified_text = modified_texts_nested_1[int(prompt_idx_str)][cand_summary['candidate_original_index']]
                print(f"  Candidate {cand_summary['candidate_original_index']} ({cand_summary['count']} operations):")
                print(f"    Original: \"{original_text}\"")
                print(f"    Modified: \"{modified_text}\"")
                for op in cand_summary['operations']:
                    print(f"      {op['operation']}: {op['details']}")
        
        if not changed_1 and modified_texts_nested_1:
             print("\nModified texts (even if no ops logged, structure shown):")
             for i, cand_list in enumerate(modified_texts_nested_1):
                 print(f"  Prompt {i}:")
                 for j, text in enumerate(cand_list):
                     print(f"    Candidate {j}: \"{text}\"")

        if "error" in ops_info_1:
             print(f"Error during processing: {ops_info_1['error']}")

    except Exception as e:
        print(f"Error in Example 1 processing: {str(e)}")
    print("-" * 30)

    # Test case 2: Multiple prompts, varying number of candidates
    test_gen_candidates_2 = [
        ["Cardiomegaly is noted.", "The heart appears enlarged. Lungs are clear."], # For GT 0
        ["Pleural effusion on the right.", "No acute findings."]  # For GT 1
    ]
    test_gt_texts_2 = [
        "Heart size is normal. Lungs are clear.", # GT 0
        "No pleural effusion." # GT 1
    ]

    print("\nTesting RaTEScore client with example 2 (2 prompts, mixed candidates):")
    print(f"Using server URL: {server_url_main}")
    print("Testing new operation priority: replace_fp_complex -> delete -> add")
    try:
        # Test with index_prompt = [0] to process only the first prompt
        modified_texts_nested_2, changed_2, ops_info_2 = process_texts_with_ratescore(
            test_gen_candidates_2, test_gt_texts_2, 
            num_operations=2,
            index_prompt=[0], # Process only the first prompt (index 0)
            server_url=server_url_main
        )
        
        print("\nResults for Example 2 (processing only prompt 0):")
        print(f"Overall text modified: {changed_2}")
        print(f"Total operations: {ops_info_2.get('total_operations', 0)}")

        for prompt_idx_str, prompt_details in ops_info_2.get("per_prompt", {}).items():
            print(f"\nPrompt {prompt_idx_str} (Total ops: {prompt_details['total_ops_for_prompt']}):")
            for cand_summary in prompt_details["candidates_summary"]:
                original_text = test_gen_candidates_2[int(prompt_idx_str)][cand_summary['candidate_original_index']]
                modified_text = modified_texts_nested_2[int(prompt_idx_str)][cand_summary['candidate_original_index']]
                print(f"  Candidate {cand_summary['candidate_original_index']} ({cand_summary['count']} operations):")
                print(f"    Original: \"{original_text}\"")
                print(f"    Modified: \"{modified_text}\"")
                for op in cand_summary['operations']:
                    print(f"      {op['operation']}: {op['details']}")
        
        print("\nFull structure of modified_texts_nested_2:")
        for i, cand_list in enumerate(modified_texts_nested_2):
            print(f"  Prompt {i} (Original GT: \"{test_gt_texts_2[i]}\"):")
            for j, text in enumerate(cand_list):
                print(f"    Candidate {j} (Original: \"{test_gen_candidates_2[i][j]}\"): \"{text}\"")

        if "error" in ops_info_2:
             print(f"Error during processing: {ops_info_2['error']}")
             
    except Exception as e:
        print(f"Error in Example 2 processing: {str(e)}")
    
    print("-" * 30)
    print("Test case 3: Empty paragraph fallback")
    # print("Note: This occurs when CLIENT-side processing results in empty text after operations")
    # print("Server only provides analysis - empty text happens when delete operations remove all sentences")
    # print("The fallback logic will activate if modified_text.strip() becomes empty and false_negatives exist")
    # print("Operation logged as: 'add_fn_empty_fallback'")
    
    # print("\nExample scenarios that would trigger fallback:")
    
    # print("\n--- Scenario 1: All sentences are false positives ---")
    # print("Generated text: 'Patient has pneumonia. Heart is enlarged. Pleural effusion present.'")
    # print("Ground truth: 'Patient has clear lungs. Heart is normal.'")
    # print("Server analysis would return:")
    # print("  false_positives: [")
    # print("    {'sentence': 'Patient has pneumonia.', ...},")
    # print("    {'sentence': 'Heart is enlarged.', ...},")
    # print("    {'sentence': 'Pleural effusion present.', ...}")
    # print("  ]")
    # print("  false_negatives: [")
    # print("    {'sentence': 'Patient has clear lungs.', ...},")
    # print("    {'sentence': 'Heart is normal.', ...}")
    # print("  ]")
    # print("Processing steps:")
    # print("  1. replace_fp_complex fails (low similarity)")
    # print("  2. delete 'Patient has pneumonia.' → 'Heart is enlarged. Pleural effusion present.'")
    # print("  3. delete 'Heart is enlarged.' → 'Pleural effusion present.'")
    # print("  4. delete 'Pleural effusion present.' → '' (EMPTY!)")
    # print("  5. FALLBACK: Add 'Patient has clear lungs.' → 'Patient has clear lungs.'")
    
    # print("\n--- Scenario 2: Single sentence completely wrong ---")
    # print("Generated text: 'Patient has brain tumor.'")
    # print("Ground truth: 'Patient has normal brain MRI.'")
    # print("Server analysis would return:")
    # print("  false_positives: [{'sentence': 'Patient has brain tumor.', ...}]")
    # print("  false_negatives: [{'sentence': 'Patient has normal brain MRI.', ...}]")
    # print("Processing steps:")
    # print("  1. replace_fp_complex fails (no similar reference sentence)")
    # print("  2. delete 'Patient has brain tumor.' → '' (EMPTY!)")
    # print("  3. FALLBACK: Add 'Patient has normal brain MRI.' → 'Patient has normal brain MRI.'")
    
    # print("\n--- Scenario 3: Very short generated text with errors ---")
    # print("Generated text: 'Cardiomegaly noted.'")
    # print("Ground truth: 'Heart size is normal. No acute findings.'")
    # print("Server analysis would return:")
    # print("  false_positives: [{'sentence': 'Cardiomegaly noted.', ...}]")
    # print("  false_negatives: [")
    # print("    {'sentence': 'Heart size is normal.', ...},")
    # print("    {'sentence': 'No acute findings.', ...}")
    # print("  ]")
    # print("Processing steps:")
    # print("  1. replace_fp_complex fails (threshold not met)")
    # print("  2. delete 'Cardiomegaly noted.' → '' (EMPTY!)")
    # print("  3. FALLBACK: Add 'Heart size is normal.' → 'Heart size is normal.'")
    
    # print("\n--- When fallback does NOT activate ---")
    # print("- No false_negatives available to add")
    # print("- Text is not empty after operations")
    # print("- replace_fp_complex or regular operations succeed")
    
    # print("\nThe fallback ensures the system never returns completely empty text!")
    # print("Without fallback: '' (empty string)")
    # print("With fallback: At least some correct content from ground truth")
    
    # print("\n" + "="*60)
    print("REAL-WORLD EXAMPLE WITH ACTUAL SERVER CALL:")
    print("="*60)
    
    # Real example from user
    real_gen_candidates = [["No significant findings."]]
    real_gt_texts = ["Chronic changes in pulmonary parenchyma. Calcified aortic atheromatosis. Aortic elongation. No other relevant findings."]
    
    print("Generated text: 'No significant findings.'")
    print("Ground truth: 'Chronic changes in pulmonary parenchyma. Calcified aortic atheromatosis. Aortic elongation. No other relevant findings.'")
    print(f"Using server URL: {server_url_main}")
    print("Operation priority: ['replace_fp_complex', 'delete', 'add']")
    
    try:
        print("\n--- CALLING SERVER FOR ANALYSIS ---")
        modified_texts_real, changed_real, ops_info_real = process_texts_with_ratescore(
            real_gen_candidates, real_gt_texts, 
            num_operations=3, # Allow up to 3 operations to demonstrate fallback
            index_prompt=[0], # Process this specific example
            server_url=server_url_main
        )
        
        print("\n--- RESULTS ---")
        print(f"Text was modified: {changed_real}")
        print(f"Total operations performed: {ops_info_real.get('total_operations', 0)}")
        
        if "error" in ops_info_real:
            print(f"ERROR: {ops_info_real['error']}")
        else:
            # Show detailed operations
            for prompt_idx_str, prompt_details in ops_info_real.get("per_prompt", {}).items():
                print(f"\nPrompt {prompt_idx_str} processing details:")
                print(f"  Total successful operations for this prompt: {prompt_details['total_ops_for_prompt']}")
                
                for cand_summary in prompt_details["candidates_summary"]:
                    original_text = real_gen_candidates[int(prompt_idx_str)][cand_summary['candidate_original_index']]
                    modified_text = modified_texts_real[int(prompt_idx_str)][cand_summary['candidate_original_index']]
                    
                    print(f"\n  ORIGINAL TEXT: \"{original_text}\"")
                    print(f"  FINAL OUTPUT:  \"{modified_text}\"")
                    
                    total_attempts = cand_summary['count']
                    successful_ops = cand_summary.get('successful_count', total_attempts)
                    failed_attempts = total_attempts - successful_ops
                    
                    if failed_attempts > 0:
                        print(f"  Operations attempted: {total_attempts} (successful: {successful_ops}, failed: {failed_attempts})")
                    else:
                        print(f"  Operations performed: {total_attempts}")
                    
                    for i, op in enumerate(cand_summary['operations'], 1):
                        op_status = " [FAILED]" if op['operation'].endswith('_failed') else ""
                        print(f"    {i}. Operation: {op['operation']}{op_status}")
                        if 'details' in op:
                            details = op['details']
                            if 'old_sentence' in details and 'new_sentence' in details:
                                print(f"       Replaced: \"{details['old_sentence']}\"")
                                print(f"       With:     \"{details['new_sentence']}\"")
                                if 'similarity_score' in details:
                                    print(f"       Similarity score: {details['similarity_score']:.3f}")
                            elif 'deleted_sentence' in details:
                                print(f"       Deleted: \"{details['deleted_sentence']}\"")
                            elif 'added_sentence' in details:
                                print(f"       Added: \"{details['added_sentence']}\"")
                                if op['operation'] == 'add_fn_empty_fallback':
                                    print(f"       *** FALLBACK ACTIVATED - Text was empty! ***")
                            elif 'attempted_sentence' in details:
                                print(f"       Attempted to replace: \"{details['attempted_sentence']}\"")
                                print(f"       Reason failed: {details['reason']}")
                                print(f"       → Proceeding to next operation type")
            
            # Summary
            if not changed_real:
                print("\nNo operations were performed - text remained unchanged")
            else:
                final_result = modified_texts_real[0][0]
                if not final_result.strip():
                    print("\nWARNING: Final text is empty!")
                else:
                    print(f"\nSUCCESS: Final text contains {len(final_result.split('.'))-1} sentences")
                    
    except Exception as e:
        print(f"ERROR during real example processing: {str(e)}")
        print("Make sure the RaTEScore server is running and accessible!")
    
    print("\nThis real example demonstrates:")
    print("  1. Actual server analysis of the generated vs ground truth text")
    print("  2. Step-by-step operations performed by the client")
    print("  3. Whether fallback mechanism activates")
    print("  4. Final output that would be returned to the user")
    
    print("Note: These tests require the RaTEScore server to be running and accessible.") 