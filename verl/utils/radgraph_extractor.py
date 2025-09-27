import copy
import re
import random # Added for randomization

def annotations_to_text(annotations):
    """Convert annotations dictionary to a list of text paragraphs."""
    paragraphs = []
    # Sort by text_id (which are string numbers '0', '1', ...) to maintain order
    # Convert to int for sorting, then back to string if keys are always numeric strings
    # Or rely on string sort if '10' vs '2' is not an issue (Python's string sort is lexicographical)
    # For '0', '1', ..., '9', '10', sorting string keys directly works.
    for text_id in sorted(annotations.keys(), key=lambda k: int(k)):
        paragraphs.append(annotations[text_id]['text'])
    return paragraphs

def get_entity_containing_sentences(text, entity_tokens=None):
    """
    Find the sentence that contains the entity tokens.
    If entity_tokens is None or empty, returns all sentences.
    """
    sentences = []
    current = ""
    if not text: # Handle empty text input
        return [], None

    # Basic sentence splitting
    for char_idx, char in enumerate(text):
        current += char
        if char in ['.', '!', '?']:
            if current.strip():
                sentences.append(current.strip())
            current = ""
        elif char_idx == len(text) - 1 and current.strip(): # Handle end of text
            sentences.append(current.strip())
            current = ""

    if not sentences and text.strip(): # If no standard punctuation but text exists
        sentences.append(text.strip())

    if not entity_tokens: # If no entity tokens, return all found sentences
        return sentences, None

    entity_sentence_found = None
    for sentence in sentences:
        # Use re.escape to handle special characters in entity_tokens
        if re.search(r"\b" + re.escape(entity_tokens.lower()) + r"\b", sentence.lower(), flags=re.IGNORECASE):
            entity_sentence_found = sentence
            break
            
    return sentences, entity_sentence_found

def compare_annotations(gen_annotation_single, gt_annotation_single):
    """Compare a single generated annotation with its corresponding single ground truth annotation."""
    fp_entities = {}
    fn_entities = {}
    contradicting_entities = {}
    
    # gt_annotation_single is now the direct annotation object for the corresponding GT
    if not gt_annotation_single or not isinstance(gt_annotation_single, dict):
        # If corresponding GT is empty or not a dict, all gen entities are FPs
        if gen_annotation_single and 'entities' in gen_annotation_single:
            for entity_id, entity in gen_annotation_single['entities'].items():
                if 'definitely absent' in entity['label'] or 'definitely present' in entity['label']:
                    fp_entities[entity_id] = entity
        return fp_entities, fn_entities, contradicting_entities

    gt_entity_map = {} # Maps token to (entity_id, entity_detail, label_type)
    if 'entities' in gt_annotation_single:
        for entity_id, entity in gt_annotation_single['entities'].items():
            if 'definitely absent' in entity['label'] or 'definitely present' in entity['label']:
                token = entity['tokens']
                label_type = 'absent' if 'definitely absent' in entity['label'] else 'present'
                entity_detail_for_map = copy.deepcopy(entity)
                _sents, entity_sent = get_entity_containing_sentences(gt_annotation_single['text'], token)
                entity_detail_for_map['sentence'] = entity_sent 
                gt_entity_map[token.lower()] = (entity_id, entity_detail_for_map, label_type)
            
    gen_tokens_present_or_absent = set()
    if gen_annotation_single and 'entities' in gen_annotation_single:
        for entity in gen_annotation_single['entities'].values():
            if 'definitely absent' in entity['label'] or 'definitely present' in entity['label']:
                gen_tokens_present_or_absent.add(entity['tokens'])

    if gen_annotation_single and 'entities' in gen_annotation_single:
        for entity_id, entity in gen_annotation_single['entities'].items():
            if 'definitely absent' in entity['label'] or 'definitely present' in entity['label']:
                token = entity['tokens']
                gen_label_type = 'absent' if 'definitely absent' in entity['label'] else 'present'
                
                if token.lower() in gt_entity_map:
                    _, gt_entity_detail, gt_label_type = gt_entity_map[token.lower()]
                    if gen_label_type != gt_label_type:
                        contradicting_entities[entity_id] = {
                            'gen_entity': entity,
                            'gt_entity': gt_entity_detail, 
                            'gen_label_type': gen_label_type,
                            'gt_label_type': gt_label_type,
                            'token': token
                        }
                else: 
                    fp_entities[entity_id] = entity

    processed_gt_sentences_for_fn = set()
    if 'entities' in gt_annotation_single:
        for entity_id, entity_detail in gt_annotation_single['entities'].items():
            token = entity_detail['tokens']
            if ('definitely absent' in entity_detail['label'] or 'definitely present' in entity_detail['label']) \
            and token not in gen_tokens_present_or_absent:
                gt_sentence_for_fn = entity_detail.get('sentence')
                if not gt_sentence_for_fn:
                    _sents_fn, gt_sentence_for_fn = get_entity_containing_sentences(gt_annotation_single['text'], token)

                if gt_sentence_for_fn and gt_sentence_for_fn not in processed_gt_sentences_for_fn:
                    fn_entities[entity_id] = {
                        'tokens': token,
                        'label': entity_detail['label'],
                        'sentence': gt_sentence_for_fn 
                    }
                    processed_gt_sentences_for_fn.add(gt_sentence_for_fn)
        
    return fp_entities, fn_entities, contradicting_entities

def replace_contradicting_sentence(
    gt_annotation_single, # Now takes single GT annotation for context
    contradiction_detail, 
    paragraph_text_to_modify
):
    token_to_replace = contradiction_detail['token']
    gt_entity_sentence = contradiction_detail['gt_entity'].get('sentence')

    _, gen_entity_sentence = get_entity_containing_sentences(paragraph_text_to_modify, token_to_replace)
    
    if not gen_entity_sentence or not gt_entity_sentence:
        return paragraph_text_to_modify, None 

    new_paragraph_text = paragraph_text_to_modify.replace(gen_entity_sentence, gt_entity_sentence, 1)
    
    replacement_info = {
        'token': token_to_replace,
        'old_sentence': gen_entity_sentence,
        'new_sentence': gt_entity_sentence,
        'old_label': contradiction_detail['gen_entity']['label'],
        'new_label': contradiction_detail['gt_entity']['label']
    }
    return new_paragraph_text, replacement_info

def delete_fp_sentence(fp_entity_detail, paragraph_text_to_modify):
    token_to_delete = fp_entity_detail['tokens']

    if not paragraph_text_to_modify.strip():
        return "", None 

    all_sentences_in_paragraph, entity_sentence_to_remove = get_entity_containing_sentences(
        paragraph_text_to_modify, token_to_delete
    )

    if not entity_sentence_to_remove:
        return paragraph_text_to_modify, None

    remaining_sentences = []
    actual_deleted_sentence = None
    found_and_removed = False
    for s in all_sentences_in_paragraph:
        if s == entity_sentence_to_remove and not found_and_removed:
            actual_deleted_sentence = s 
            found_and_removed = True
        else:
            remaining_sentences.append(s)
    
    if not found_and_removed:
        return paragraph_text_to_modify, None 

    new_paragraph_text = " ".join(remaining_sentences).strip()
    return new_paragraph_text, actual_deleted_sentence

def add_fn_sentence(fn_entity_detail, paragraph_text_to_modify):
    fn_sentence_to_add = fn_entity_detail.get('sentence')

    if not fn_sentence_to_add:
        return paragraph_text_to_modify, None

    if paragraph_text_to_modify.strip():
        new_paragraph_text = paragraph_text_to_modify.strip() + " " + fn_sentence_to_add
    else:
        new_paragraph_text = fn_sentence_to_add
        
    return new_paragraph_text, fn_sentence_to_add

def replace_fp_with_gt_present(fp_entity_detail, gt_annotations, paragraph_text_to_modify):
    """
    Replace a false positive sentence with a sentence from ground truth that contains a 'definitely present' entity.
    
    Args:
        fp_entity_detail: Details of the false positive entity in the generated text
        gt_annotations: Ground truth annotations (this is current_gt_anno_single for the current paragraph)
        paragraph_text_to_modify: Current paragraph text to modify
    
    Returns:
        tuple: (modified paragraph text, replacement info dict or None if no replacement occurred)
    """
    token_to_replace = fp_entity_detail['tokens']
    
    # Get the sentence containing the FP entity
    _sents, fp_sentence = get_entity_containing_sentences(paragraph_text_to_modify, token_to_replace)
    
    if not fp_sentence:
        return paragraph_text_to_modify, None
    
    # gt_annotations is current_gt_anno_single here, the single annotation object for the current GT text
    gt_text_data = gt_annotations 

    candidate_gt_replacements = [] # Collect all suitable GT sentences

    if gt_text_data and isinstance(gt_text_data, dict) and 'entities' in gt_text_data:
        # Iterate through entities marked as 'definitely present' in the current GT annotation
        for _entity_id, entity in gt_text_data['entities'].items():
            if 'definitely present' in entity['label']:
                token = entity['tokens']
                # Make sure the sentence is pre-fetched or fetch it now
                entity_sent = entity.get('sentence')
                if not entity_sent and gt_text_data.get('text'): # Ensure GT text exists
                    _sents_gt, entity_sent = get_entity_containing_sentences(gt_text_data['text'], token)
                
                # Add to candidates if the sentence is valid and not already in the target paragraph
                if entity_sent and entity_sent not in paragraph_text_to_modify:
                    candidate_gt_replacements.append({'sentence': entity_sent, 'token': token})
    
    if not candidate_gt_replacements: # No suitable GT sentence found
        return paragraph_text_to_modify, None
        
    # Randomly choose one of the suitable GT sentences for replacement
    selected_replacement = random.choice(candidate_gt_replacements)
    replacement_sentence = selected_replacement['sentence']
    replacement_token = selected_replacement['token']
    
    # Replace the FP sentence with the chosen GT sentence
    new_paragraph_text = paragraph_text_to_modify.replace(fp_sentence, replacement_sentence, 1)
    
    replacement_info = {
        'fp_token': token_to_replace,
        'gt_token': replacement_token,
        'old_sentence': fp_sentence,
        'new_sentence': replacement_sentence,
    }
    
    return new_paragraph_text, replacement_info

def process_annotations(gen_annotations, gt_annotations, num_operations=1, index_paragraph=None, operation_enable=None):
    if operation_enable is None:
        operation_enable = ["replace", "delete", "add"]  # Default: all operations enabled
    
    # Handle case where gen_annotations might be empty (e.g. from server if generations list was empty)
    if not gen_annotations:
        modified_paras_final = []
        final_ops_info_empty_gen = {
            'total_operations': 0,
            'per_paragraph': {}
        }
        # Potentially, if gt_annotations is not empty, we could try to "add" all GT content
        # But current logic for empty gen focuses on returning empty results or minimal additions if any.
        # For simplicity, if gen_annotations is empty, return empty modified_paras.
        # The original code had a section for "if not gen_annotations: _fp, fn_only, _ctr = compare_annotations({}, gt_annotations)"
        # This would require compare_annotations to handle a global gt_annotations map.
        # The refactor makes compare_annotations expect a single gen and single gt.
        # To handle empty gen_annotations while having gt_annotations, process_annotations would need special logic here.
        # For now, keeping it simple: if no gen_annotations, no operations, return empty list.
        # The old block for handling empty gen annotations and only FNs needs to be adapted if we want to support it.
        # The previous empty gen handling was: 
        # _fp, fn_only, _ctr = compare_annotations({}, gt_annotations)
        # modified_paras = [""] 
        # ... etc. This implies compare_annotations was taking the global gt_annotations map.
        # With the new per-paragraph comparison, if gen_annotations is empty, there are no paragraphs to process.
        return modified_paras_final, False, final_ops_info_empty_gen

    change_flag = False
    modified_paragraphs = annotations_to_text(gen_annotations) # Initial texts
    
    processed_tokens_global = set() # To avoid reusing same token across different paragraphs for certain ops if needed (currently not used like this)
    processed_sentences_global = set() # Same as above
    
    # Determine which paragraphs to process based on index_paragraph
    num_gen_paragraphs = len(modified_paragraphs)
    if index_paragraph is None or (isinstance(index_paragraph, list) and len(index_paragraph) == 1 and index_paragraph[0] == -1):
        # Shuffle all paragraph indices if processing all, as per user's previous modification
        all_paragraph_indices = list(range(num_gen_paragraphs))
        random.shuffle(all_paragraph_indices)
        # Select approximately half of the paragraphs (or as per user modification)
        num_to_select = len(all_paragraph_indices) // 2
        allowed_paragraphs_indices_list = all_paragraph_indices[:num_to_select]
        # allowed_paragraphs_indices_list = all_paragraph_indices # Process all if -1, but shuffled
    else:
        allowed_paragraphs_indices_list = [idx for idx in index_paragraph if 0 <= idx < num_gen_paragraphs]
    
    # Initialize operations log for all potential paragraphs based on gen_annotations
    # Keys for paragraph_operations should match keys in gen_annotations (e.g., '0', '1', ...)
    paragraph_operations = {str(i): {'count': 0, 'operations': []} for i in range(num_gen_paragraphs)}
    
    # Iterate through the selected (and possibly shuffled) paragraph indices
    for para_idx in allowed_paragraphs_indices_list:
        text_id_gen = str(para_idx) # Key for gen_annotations and modified_paragraphs
        
        # Get the specific generated annotation for this paragraph
        current_gen_anno_single = gen_annotations.get(text_id_gen)
        if not current_gen_anno_single:
            continue # Should not happen if para_idx is from range(len(modified_paragraphs))

        # Get the corresponding ground truth annotation
        # gt_annotations is a map like {"0": anno, "1": anno}. We need the one for this text_id_gen.
        current_gt_anno_single = gt_annotations.get(text_id_gen) # Assuming keys match
        # If no specific GT for this gen_text_id, or if GT is empty, operations might be limited
        # compare_annotations handles empty/None current_gt_anno_single

        # Perform comparison for this specific paragraph
        fp_entities_para, fn_entities_para, contradicting_entities_para = \
            compare_annotations(current_gen_anno_single, current_gt_anno_single)

        # Local processed items for this paragraph to ensure one op per item type within the num_operations budget for this paragraph
        processed_tokens_para = set()
        processed_sentences_para = set()

        if paragraph_operations[text_id_gen]['count'] >= num_operations:
            continue
            
        while paragraph_operations[text_id_gen]['count'] < num_operations:
            current_paragraph_text = modified_paragraphs[para_idx] # Get potentially modified text
            operation_performed_this_cycle = False

            # Replacement operation (contradictions)
            if "replace" in operation_enable:
                candidate_contradictions = []
                # contradicting_entities_para is already specific to this paragraph
                for _entity_id_c, contradiction_c in contradicting_entities_para.items():
                    token_c = contradiction_c['token']
                    gt_sentence_for_replacement = contradiction_c['gt_entity'].get('sentence')
                    _sents, gen_entity_sentence_c = get_entity_containing_sentences(current_paragraph_text, token_c)
                    if token_c not in processed_tokens_para and \
                       gen_entity_sentence_c and \
                       (gt_sentence_for_replacement and gt_sentence_for_replacement not in processed_sentences_para):
                        candidate_contradictions.append(contradiction_c)
                
                if candidate_contradictions:
                    selected_contradiction_detail = random.choice(candidate_contradictions)
                    new_para_text, replacement_info = replace_contradicting_sentence(
                        current_gt_anno_single, selected_contradiction_detail, current_paragraph_text)
                    if replacement_info:
                        modified_paragraphs[para_idx] = new_para_text
                        change_flag = True
                        paragraph_operations[text_id_gen]['count'] += 1
                        paragraph_operations[text_id_gen]['operations'].append({'operation': 'replace', 'details': replacement_info})
                        processed_tokens_para.add(selected_contradiction_detail['token'])
                        processed_sentences_para.add(replacement_info['new_sentence'])
                        processed_sentences_para.add(replacement_info['old_sentence'])
                        operation_performed_this_cycle = True
                        current_paragraph_text = new_para_text 
            
            if operation_performed_this_cycle: continue
            
            # Replace FP sentences with GT sentences containing present tokens
            if "replace" in operation_enable:
                candidate_fp_for_replacement = []
                # fp_entities_para is specific to this paragraph
                for _entity_id_fp, entity_fp in fp_entities_para.items():
                    token_fp = entity_fp['tokens']
                    _sents, offending_sent = get_entity_containing_sentences(current_paragraph_text, token_fp)
                    if token_fp not in processed_tokens_para and \
                       offending_sent and \
                       offending_sent not in processed_sentences_para: 
                        candidate_fp_for_replacement.append(entity_fp)
                
                if candidate_fp_for_replacement:
                    selected_fp_entity_detail_for_replacement = random.choice(candidate_fp_for_replacement)
                    new_para_text, replacement_info = replace_fp_with_gt_present(
                        selected_fp_entity_detail_for_replacement, current_gt_anno_single, current_paragraph_text)
                    if replacement_info:
                        modified_paragraphs[para_idx] = new_para_text
                        change_flag = True
                        paragraph_operations[text_id_gen]['count'] += 1
                        paragraph_operations[text_id_gen]['operations'].append({
                            'operation': 'replace_fp', 
                            'details': replacement_info
                        })
                        processed_tokens_para.add(selected_fp_entity_detail_for_replacement['tokens'])
                        processed_sentences_para.add(replacement_info['new_sentence'])
                        processed_sentences_para.add(replacement_info['old_sentence'])
                        operation_performed_this_cycle = True
                        current_paragraph_text = new_para_text 
            
            if operation_performed_this_cycle: continue

            # Delete operation (false positives)
            if "delete" in operation_enable:
                candidate_fps_for_deletion = []
                # fp_entities_para is specific to this paragraph
                for _entity_id_fp, entity_fp in fp_entities_para.items():
                    token_fp = entity_fp['tokens']
                    _sents, offending_sent = get_entity_containing_sentences(current_paragraph_text, token_fp)
                    if token_fp not in processed_tokens_para and \
                       offending_sent and \
                       offending_sent not in processed_sentences_para: 
                        candidate_fps_for_deletion.append(entity_fp)
                
                if candidate_fps_for_deletion:
                    selected_fp_entity_detail = random.choice(candidate_fps_for_deletion)
                    new_para_text, deleted_sentence_str = delete_fp_sentence(selected_fp_entity_detail, current_paragraph_text)
                    if deleted_sentence_str is not None: 
                        modified_paragraphs[para_idx] = new_para_text
                        change_flag = True
                        paragraph_operations[text_id_gen]['count'] += 1
                        paragraph_operations[text_id_gen]['operations'].append({
                            'operation': 'delete', 
                            'details': {'deleted_sentence': deleted_sentence_str, 'token': selected_fp_entity_detail['tokens']}
                        })
                        processed_tokens_para.add(selected_fp_entity_detail['tokens'])
                        processed_sentences_para.add(deleted_sentence_str)
                        operation_performed_this_cycle = True
                        current_paragraph_text = new_para_text 

                        if not current_paragraph_text.strip(): 
                            if current_gt_anno_single and current_gt_anno_single.get('text'): 
                                gt_text_content_para = current_gt_anno_single['text']
                                gt_sentences_for_filling, _ = get_entity_containing_sentences(gt_text_content_para, None)
                                if gt_sentences_for_filling:
                                    sentence_to_fill_report = random.choice(gt_sentences_for_filling) 
                                    modified_paragraphs[para_idx] = sentence_to_fill_report
                                    current_paragraph_text = sentence_to_fill_report
                                    last_op_entry = paragraph_operations[text_id_gen]['operations'][-1]
                                    last_op_entry['details']['operation_note'] = 'Paragraph emptied by delete, filled with a GT sentence from its corresponding GT.'
                                    last_op_entry['details']['filled_with_gt_sentence'] = sentence_to_fill_report
            
            if operation_performed_this_cycle: continue

            # Add operation (false negatives)
            if "add" in operation_enable:
                candidate_fns_for_addition = []
                # fn_entities_para is specific to this paragraph
                for _entity_id_fn, entity_fn in fn_entities_para.items():
                    token_fn = entity_fn['tokens']
                    sentence_fn = entity_fn['sentence'] 
                    if token_fn not in processed_tokens_para and sentence_fn and sentence_fn not in processed_sentences_para:
                        candidate_fns_for_addition.append(entity_fn)
                
                if candidate_fns_for_addition:
                    selected_fn_entity_detail = random.choice(candidate_fns_for_addition)
                    new_para_text, added_sentence_str = add_fn_sentence(selected_fn_entity_detail, current_paragraph_text)
                    if added_sentence_str:
                        modified_paragraphs[para_idx] = new_para_text
                        change_flag = True
                        paragraph_operations[text_id_gen]['count'] += 1
                        paragraph_operations[text_id_gen]['operations'].append({
                            'operation': 'add', 
                            'details': {'added_sentence': added_sentence_str, 'token': selected_fn_entity_detail['tokens']}
                        })
                        processed_tokens_para.add(selected_fn_entity_detail['tokens'])
                        processed_sentences_para.add(added_sentence_str)
                        operation_performed_this_cycle = True
                        current_paragraph_text = new_para_text 
            
            if not operation_performed_this_cycle: 
                break 
    
    final_operation_info = {
        'total_operations': sum(paragraph_operations[k]['count'] for k in paragraph_operations),
        'per_paragraph': {k_idx_str: v for k_idx_str, v in paragraph_operations.items() if v['count'] > 0}
    }
    return modified_paragraphs, change_flag, final_operation_info

def update_annotations_from_text(original_annotations_structure_template, modified_paragraphs_list):
    modified_annotations = {}
    original_keys = []
    if original_annotations_structure_template:
        original_keys = sorted(original_annotations_structure_template.keys(), key=lambda k: int(k))

    for i, para_text in enumerate(modified_paragraphs_list):
        current_key = str(i) 
        if original_annotations_structure_template and i < len(original_keys):
            current_key = original_keys[i]
        
        new_entity_dict = {}
        if original_annotations_structure_template and current_key in original_annotations_structure_template:
            if original_annotations_structure_template[current_key]['text'] == para_text:
                new_entity_dict = copy.deepcopy(original_annotations_structure_template[current_key].get('entities', {}))
        
        data_source_val = None
        data_split_val = 'inference'
        if original_annotations_structure_template and current_key in original_annotations_structure_template:
            data_source_val = original_annotations_structure_template[current_key].get('data_source')
            data_split_val = original_annotations_structure_template[current_key].get('data_split', 'inference')


        modified_annotations[current_key] = {
            'text': para_text,
            'entities': new_entity_dict, 
            'data_source': data_source_val,
            'data_split': data_split_val
        }
    return modified_annotations

if __name__ == "__main__":
    example_gt_annotations = {
        '0': {
            'text': 'Normal chest x-ray. No pneumonia. Hiatal hernia is not visualized.',
            'entities': {
                '1': {'tokens': 'normal', 'label': 'Observation::definitely present'},
                '2': {'tokens': 'chest', 'label': 'Anatomy::definitely present'},
                '3': {'tokens': 'x-ray', 'label': 'Observation::definitely present'},
                '4': {'tokens': 'pneumonia', 'label': 'Observation::definitely absent'},
                '5': {'tokens': 'hiatal hernia', 'label': 'Observation::definitely absent'}
            }
        }
    }

    example_gen_annotations = {
        '0': {
            'text': 'No evidence of pneumonia. Moderate hiatal hernia present.',
            'entities': {
                '1': {'tokens': 'pneumonia', 'label': 'Observation::definitely absent'}, # Matches GT label, but token "hiatal hernia" contradicts
                '2': {'tokens': 'hiatal hernia', 'label': 'Observation::definitely present'} # Contradicts GT (absent)
            }
        },
        '1': {
            'text': 'The cardiomediastinal silhouette is normal.', # "normal" is FN, "cardiomediastinal silhouette" is FP
            'entities': {
                '1': {'tokens': 'cardiomediastinal silhouette', 'label': 'Anatomy::definitely present'}, # Not in simple GT
                '2': {'tokens': 'normal', 'label': 'Observation::definitely present'} # Matches GT token "normal"
            }
        },
        '2': {
            'text': 'Clear lungs bilaterally. No pleural effusion.', # FPs if not in GT
            'entities': {
                '1': {'tokens': 'lungs', 'label': 'Anatomy::definitely present'},
                '2': {'tokens': 'clear', 'label': 'Observation::definitely present'},
                '3': {'tokens': 'pleural effusion', 'label': 'Observation::definitely absent'}
            }
        }
    }

    def run_example(gen_annotations, gt_annotations, example_name, num_operations=1, index_paragraph=None, operation_enable=None):
        print(f"\n\n{'=' * 60}")
        print(f"Example: {example_name} with num_operations={num_operations}, index_paragraph={index_paragraph}, operation_enable={operation_enable}")
        print("Ground Truth:")
        if gt_annotations and list(gt_annotations.keys()):
            for gt_text_id in sorted(gt_annotations.keys()):
                print(f"  GT {gt_text_id}: {gt_annotations[gt_text_id]['text']}")
        else:
            print("  (No ground truth provided or empty)")


        print("\nOriginal Generated Text:")
        if gen_annotations:
            for i_str_key in sorted(gen_annotations.keys(), key=lambda k: int(k)):
                print(f"  Paragraph {i_str_key}: {gen_annotations[i_str_key]['text']}")
        else:
            print("  (No generated text)")


        modified_text_list, change_flag, operation_info_dict = process_annotations(
            gen_annotations, gt_annotations, num_operations=num_operations, 
            index_paragraph=index_paragraph, operation_enable=operation_enable
        )

        if change_flag:
            print(f"\nTotal changes performed: {operation_info_dict['total_operations']}")
            if operation_info_dict.get('per_paragraph'):
                 for text_id_op, details_op in sorted(operation_info_dict['per_paragraph'].items(), key=lambda item: int(item[0])):
                    print(f"\nParagraph {text_id_op} ({details_op['count']} operations):")
                    for op_info in details_op['operations']:
                        # Handle different types of operations for display clarity
                        if op_info['operation'] == 'replace':
                            print(f"  replace: Token '{op_info['details']['token']}' - Replaced '{op_info['details']['old_sentence']}' with '{op_info['details']['new_sentence']}'")
                        elif op_info['operation'] == 'replace_fp':
                            print(f"  replace_fp: FP Token '{op_info['details']['fp_token']}' replaced with GT token '{op_info['details']['gt_token']}' - Replaced '{op_info['details']['old_sentence']}' with '{op_info['details']['new_sentence']}'")
                        elif op_info['operation'] == 'delete':
                            print(f"  delete: Token '{op_info['details']['token']}' - Deleted '{op_info['details']['deleted_sentence']}'")
                            if 'operation_note' in op_info['details']:
                                print(f"    Note: {op_info['details']['operation_note']}")
                                if 'filled_with_gt_sentence' in op_info['details']:
                                    print(f"    Filled with: '{op_info['details']['filled_with_gt_sentence']}'")
                        elif op_info['operation'] == 'add':
                            print(f"  add: Token '{op_info['details']['token']}' - Added '{op_info['details']['added_sentence']}'")
                        else:
                            print(f"  {op_info['operation']}: {op_info['details']}")
            else:
                print("  (No per-paragraph operation details logged, but changes were made)")
        else:
            print("\nNo changes were made to the text.")

        print("\nModified Text:")
        if modified_text_list:
            for i, paragraph in enumerate(modified_text_list):
                print(f"  Paragraph {i}: {paragraph}")
        else:
            print("  (No modified text produced)")
        print(f"{'=' * 60}")

    run_example(copy.deepcopy(example_gen_annotations), copy.deepcopy(example_gt_annotations), 
                "Complex Annotations, 1 op on para 0", num_operations=1, index_paragraph=[0])
    
    run_example(copy.deepcopy(example_gen_annotations), copy.deepcopy(example_gt_annotations), 
                "Complex Annotations, 2 ops on para 0, 1", num_operations=2, index_paragraph=[0, 1])

    run_example(copy.deepcopy(example_gen_annotations), copy.deepcopy(example_gt_annotations), 
                "Complex Annotations, 3 ops, all paras", num_operations=3, index_paragraph=None) 

    # Examples with controlled operations
    run_example(copy.deepcopy(example_gen_annotations), copy.deepcopy(example_gt_annotations), 
                "Only replace operations", num_operations=2, operation_enable=["replace"])
                
    run_example(copy.deepcopy(example_gen_annotations), copy.deepcopy(example_gt_annotations), 
                "Only delete operations", num_operations=2, operation_enable=["delete"])
                
    run_example(copy.deepcopy(example_gen_annotations), copy.deepcopy(example_gt_annotations), 
                "Only add operations", num_operations=2, operation_enable=["add"])
                
    run_example(copy.deepcopy(example_gen_annotations), copy.deepcopy(example_gt_annotations), 
                "Replace and delete, no add", num_operations=2, operation_enable=["replace", "delete"])

    example1_gen_annotations = { 
        '0': {
            'text': 'The lungs are clear. No pleural effusion. Extra finding here.',
            'entities': {
                '1': {'tokens': 'lungs', 'label': 'Anatomy::definitely present'}, 
                '2': {'tokens': 'clear', 'label': 'Observation::definitely present'}, 
                '3': {'tokens': 'pleural effusion', 'label': 'Observation::definitely absent'}, 
                '4': {'tokens': 'Extra finding here', 'label': 'Observation::definitely present'} 
            }
        }
    }
    run_example(copy.deepcopy(example1_gen_annotations), copy.deepcopy(example_gt_annotations), 
                "False Positives Only, 1 op", num_operations=1)
    run_example(copy.deepcopy(example1_gen_annotations), copy.deepcopy(example_gt_annotations), 
                "False Positives Only, 2 ops", num_operations=2)


    example2_gen_annotations = { 
        '0': {
            'text': 'The report is okay.', 
            'entities': {'1': {'tokens': 'okay', 'label': 'Observation::definitely present'}}
        }
    }
    run_example(copy.deepcopy(example2_gen_annotations), copy.deepcopy(example_gt_annotations), 
                "False Negatives Only, 1 op", num_operations=1)
    run_example(copy.deepcopy(example2_gen_annotations), copy.deepcopy(example_gt_annotations), 
                "False Negatives Only, 2 ops", num_operations=2)
    run_example(copy.deepcopy(example2_gen_annotations), copy.deepcopy(example_gt_annotations), 
                "False Negatives Only, 3 ops", num_operations=3)


    example3_gt_annotations = {
        '0': {
            'text': 'Normal cardiomediastinal silhouette.',
            'entities': {
                '1': {'tokens': 'normal', 'label': 'Observation::definitely present'},
                '2': {'tokens': 'cardiomediastinal silhouette', 'label': 'Anatomy::definitely present'}
            }
        }
    }
    example3_gen_annotations = {
        '0': {
            'text': 'Normal cardiomediastinal silhouette.',
            'entities': {
                '1': {'tokens': 'normal', 'label': 'Observation::definitely present'},
                '2': {'tokens': 'cardiomediastinal silhouette', 'label': 'Anatomy::definitely present'}
            }
        }
    }
    run_example(copy.deepcopy(example3_gen_annotations), copy.deepcopy(example3_gt_annotations), 
                "No Issues", num_operations=1)

    empty_report_test_gen = {
        '0': {
            'text': 'This is a false positive sentence.', 
            'entities': {'1': {'tokens': 'This is a false positive sentence', 'label': 'Observation::definitely present'}}
        }
    }
    empty_report_test_gt = { 
        '0': {
            'text': 'Ground truth sentence one. Ground truth sentence two.',
            'entities': {
                '1': {'tokens': 'Ground truth sentence one', 'label': 'Observation::definitely present'},
                '2': {'tokens': 'Ground truth sentence two', 'label': 'Observation::definitely present'}
            }
        }
    }
    run_example(copy.deepcopy(empty_report_test_gen), copy.deepcopy(empty_report_test_gt),
                "Test Comment 5: Delete leads to empty, then fill", num_operations=1)

    run_example({}, copy.deepcopy(example_gt_annotations), "Empty Gen Annotations, 2 ops", num_operations=2)
    
    run_example({}, copy.deepcopy(example_gt_annotations), 
                "Empty Gen with only add operations", num_operations=2, operation_enable=["add"])
    
    run_example({}, copy.deepcopy(example_gt_annotations), 
                "Empty Gen with only replace operations (should do nothing)", num_operations=2, operation_enable=["replace"])
    
    contradiction_then_add_gen = {
        '0': {
            'text': "Pneumonia is present. Some other findings.",
            'entities': {
                '1': {'tokens': 'Pneumonia', 'label': 'Observation::definitely present'}, # Contradicts GT 'pneumonia' (absent)
                '2': {'tokens': 'Some other findings', 'label': 'Observation::definitely present'} # FP
            }
        }
    }
    # example_gt_annotations has: 'pneumonia': absent, 'hiatal hernia': absent, 'normal chest x-ray': present
    run_example(copy.deepcopy(contradiction_then_add_gen), copy.deepcopy(example_gt_annotations),
                "Contradiction (pneumonia) then Add (hiatal hernia if FN) or Del (other findings), 2 ops", num_operations=2)
    
    # Example with specific operation sequences
    run_example(copy.deepcopy(contradiction_then_add_gen), copy.deepcopy(example_gt_annotations),
                "Replace only - fix contradictions", num_operations=2, operation_enable=["replace"])
                
    run_example(copy.deepcopy(contradiction_then_add_gen), copy.deepcopy(example_gt_annotations),
                "Delete only - remove false positives", num_operations=2, operation_enable=["delete"])

    # Add a new test example to demonstrate the new replacement functionality
    example_fp_replacement_gen = {
        '0': {
            'text': 'There is cardiomegaly. Multiple lung nodules are present.', 
            'entities': {
                '1': {'tokens': 'cardiomegaly', 'label': 'Observation::definitely present'},
                '2': {'tokens': 'lung nodules', 'label': 'Observation::definitely present'}
            }
        }
    }
    
    # Ground truth has 'normal chest' which is a present entity that doesn't match any FP in gen
    run_example(copy.deepcopy(example_fp_replacement_gen), copy.deepcopy(example_gt_annotations), 
                "FP Replacement with GT Present Entities", num_operations=2, operation_enable=["replace"])
                
    # Another example with specific FP sentences to be replaced
    example_replacement_test = {
        '0': {
            'text': 'The lungs are abnormal. The heart is enlarged. Pleural effusion is seen.',
            'entities': {
                '1': {'tokens': 'lungs', 'label': 'Anatomy::definitely present'},
                '2': {'tokens': 'abnormal', 'label': 'Observation::definitely present'}, 
                '3': {'tokens': 'heart', 'label': 'Anatomy::definitely present'},
                '4': {'tokens': 'enlarged', 'label': 'Observation::definitely present'},
                '5': {'tokens': 'pleural effusion', 'label': 'Observation::definitely present'}
            }
        }
    }
    
    # GT has normal findings to replace the abnormal ones
    run_example(copy.deepcopy(example_replacement_test), copy.deepcopy(example_gt_annotations), 
                "Complex FP Replacement Example", num_operations=3, operation_enable=["replace"])