import traceback
import json
from flask import Flask, request, current_app, jsonify
from flask_responses import json_response
import torch
import logging
import os

# Import necessary modules for RaTEScore
try:
    from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForTokenClassification
except ImportError:
    print("Please install Hugging Face transformers and PyTorch: pip install transformers torch")
    import sys
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='ratescore_debug.log',
                    filemode='a')
logger = logging.getLogger(__name__)

# --- RaTEScore Default Configurations ---
DEFAULT_NER_MODEL_NAME = "Angelakeke/RaTE-NER-Deberta"
DEFAULT_SYN_MODEL_NAME = 'FremyCompany/BioLORD-2023-C'

# Default affinity matrix (as provided in the RaTEScore calculation script)
DEFAULT_AFFINITY_MATRIX = {
    "abnormality_abnormality": 0.4276119164393705, "abnormality_anatomy": 0.6240929990607657,
    "abnormality_disease": 0.0034478181112993847, "abnormality_non-abnormality": 0.5431049700217344,
    "abnormality_non-disease": 0.27005425386213877, "anatomy_abnormality": 0.7487824274337533,
    "anatomy_anatomy": 0.2856134859160784, "anatomy_disease": 0.4592143222158069,
    "anatomy_non-abnormality": 0.02097055139911715, "anatomy_non-disease": 0.00013736314126696204,
    "disease_abnormality": 0.8396510075734789, "disease_anatomy": 0.9950209388542061,
    "disease_disease": 0.8460555030578727, "disease_non-abnormality": 0.9820689020512646,
    "disease_non-disease": 0.3789136708096537, "non-abnormality_abnormality": 0.16546764653692908,
    "non-abnormality_anatomy": 0.018670610691852826, "non-abnormality_disease": 0.719397354576018,
    "non-abnormality_non-abnormality": 0.0009357166071730684, "non-abnormality_non-disease": 0.0927333564267591,
    "non-disease_abnormality": 0.7759420231214385, "non-disease_anatomy": 0.1839139293714062,
    "non-disease_disease": 0.10073046076318157, "non-disease_non-abnormality": 0.03860183811876373,
    "non-disease_non-disease": 0.34065681486566446
}

# Convert affinity matrix keys to (TYPE, TYPE) format
PROCESSED_AFFINITY_MATRIX = {
    (k.split('_')[0].upper(), k.split('_')[1].upper()): v
    for k, v in DEFAULT_AFFINITY_MATRIX.items()
}

# Default negative classes and weight
DEFAULT_NEG_CLASS = [
    ('NON-DISEASE', 'DISEASE'), ('NON-ABNORMALITY', 'ABNORMALITY'),
    ('DISEASE', 'NON-DISEASE'), ('ABNORMALITY', 'NON-ABNORMALITY'),
    ('NON-DISEASE', 'ABNORMALITY'), ('NON-ABNORMALITY', 'DISEASE'),
    ('DISEASE', 'NON-ABNORMALITY'), ('ABNORMALITY', 'NON-DISEASE'),
]
DEFAULT_NEG_WEIGHT = 0.3612
DEFAULT_SIMILARITY_THRESHOLD = 0.6

# --- Helper Functions ---
def get_entity_containing_sentences(text, entity_tokens=None):
    """
    Find the sentence that contains the entity tokens.
    If entity_tokens is None or empty, returns all sentences.
    """
    import re
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

def cosine_similarity(tensor1, tensor2):
    """Calculate cosine similarity between two tensors or batches of tensors."""
    # Handle empty tensors to prevent errors during normalization or matmul
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        if tensor1.ndim == 1 and tensor2.ndim == 1: return torch.tensor(0.0).to(tensor1.device)
        # For batch, need to return appropriately shaped zero tensor
        if tensor1.ndim == 2 and tensor2.ndim == 2: # (N,D) vs (M,D) -> (N,M)
            return torch.zeros(tensor1.shape[0], tensor2.shape[0]).to(tensor1.device)
        if tensor1.ndim == 1 and tensor2.ndim == 2: # (D) vs (M,D) -> (M)
            return torch.zeros(tensor2.shape[0]).to(tensor1.device)
        if tensor1.ndim == 2 and tensor2.ndim == 1: # (N,D) vs (D) -> (N)
            return torch.zeros(tensor1.shape[0]).to(tensor1.device)
        return torch.tensor(0.0).to(tensor1.device) # Fallback for other unhandled empty cases

    tensor1_norm = torch.nn.functional.normalize(tensor1, p=2, dim=-1)
    tensor2_norm = torch.nn.functional.normalize(tensor2, p=2, dim=-1)
    
    if tensor1_norm.ndim == 1 and tensor2_norm.ndim == 1:
        return torch.dot(tensor1_norm, tensor2_norm)
    elif tensor1_norm.ndim == 2 and tensor2_norm.ndim == 2: 
        return torch.matmul(tensor1_norm, tensor2_norm.transpose(0, 1))
    elif tensor1_norm.ndim == 1 and tensor2_norm.ndim == 2: 
         return torch.mv(tensor2_norm, tensor1_norm) 
    elif tensor1_norm.ndim == 2 and tensor2_norm.ndim == 1: 
         return torch.mv(tensor1_norm, tensor2_norm) 
    else:
        raise ValueError(f"Unsupported tensor dimensions for cosine_similarity: {tensor1_norm.ndim} and {tensor2_norm.ndim}")

# --- RaTEScore Model Class ---
class RateScoreModels:
    """A class to load and hold RaTEScore models to avoid reloading."""
    def __init__(self, ner_model_name=DEFAULT_NER_MODEL_NAME, 
                 syn_model_name=DEFAULT_SYN_MODEL_NAME, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # NER Model
        self.ner_config = AutoConfig.from_pretrained(ner_model_name)
        # Use ner_config.id2label directly for safer ID to Label mapping
        self.idx2label = self.ner_config.id2label 
        self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name).eval().to(self.device)
        
        # Synonym/Embedding Model
        self.syn_tokenizer = AutoTokenizer.from_pretrained(syn_model_name)
        self.syn_model = AutoModel.from_pretrained(syn_model_name).eval().to(self.device)
        print(f"RaTEScore models loaded on device: {self.device}")

    def run_ner_for_text(self, text_batch):
        """Simplified NER runner for a batch of texts."""
        inputs = self.ner_tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            logits = self.ner_model(**inputs).logits
        
        predictions = torch.argmax(logits, dim=2)
        results = []
        for i in range(len(text_batch)):
            # Ensure input_ids are on CPU for convert_ids_to_tokens if it expects that
            tokens = self.ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][i].cpu())
            # Use .get for idx2label for safety, defaulting to "O" (Outside)
            predicted_labels = [self.idx2label.get(p.item(), "O") for p in predictions[i]]
            
            current_entities = []
            current_entity_tokens = []
            current_entity_label = None
            for token, label_str in zip(tokens, predicted_labels):
                if token in [self.ner_tokenizer.cls_token, self.ner_tokenizer.sep_token, self.ner_tokenizer.pad_token]:
                    continue
                
                # Standard BIESO or BIO scheme handling
                if label_str.startswith("B-"):
                    if current_entity_tokens: 
                        current_entities.append({
                            "text": self.ner_tokenizer.convert_tokens_to_string(current_entity_tokens),
                            "label": current_entity_label 
                        })
                    current_entity_tokens = [token]
                    current_entity_label = label_str[2:] 
                elif label_str.startswith("I-") and current_entity_label == label_str[2:]:
                    current_entity_tokens.append(token)
                else: # O label, or S-label, or start of new entity without B-
                    if current_entity_tokens: # Save previous entity
                        current_entities.append({
                            "text": self.ner_tokenizer.convert_tokens_to_string(current_entity_tokens),
                            "label": current_entity_label
                        })
                        current_entity_tokens = []
                        current_entity_label = None
                    
                    if label_str != "O" and not label_str.startswith("I-"): # Start of a new entity (e.g. S-TYPE or just TYPE)
                        current_entity_tokens = [token]
                        current_entity_label = label_str[2:] if label_str.startswith("S-") else label_str
            
            if current_entity_tokens: 
                current_entities.append({
                    "text": self.ner_tokenizer.convert_tokens_to_string(current_entity_tokens),
                    "label": current_entity_label
                })
            results.append(current_entities)
        return results

    def get_embeddings_for_entities(self, entity_texts):
        """Get embeddings for a list of entity texts."""
        if not entity_texts: # Should be caught by caller, but as a safeguard
            return torch.tensor([]).to(self.device)
        
        inputs = self.syn_tokenizer(entity_texts, padding=True, truncation=True, return_tensors="pt", max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.syn_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1) # Mean pooling
        return embeddings

def extract_entities_with_context(text, rs_models: RateScoreModels):
    """
    Extracts entities from text, their types, embeddings, and containing sentences.
    Returns a list of entity_info dicts.
    """
    if not text.strip():
        return []
        
    ner_results_batch = rs_models.run_ner_for_text([text]) 
    raw_ner_entities = ner_results_batch[0] if ner_results_batch else []

    valid_ner_entities_for_embedding = []
    for entity in raw_ner_entities:
        # Ensure entity text is not just whitespace and label exists
        if entity.get('text', '').strip() and entity.get('label') is not None:
            valid_ner_entities_for_embedding.append(entity)

    if not valid_ner_entities_for_embedding:
        return []

    entity_texts = [entity['text'] for entity in valid_ner_entities_for_embedding]
    # This call should now receive a non-empty list of non-empty strings
    entity_embeddings = rs_models.get_embeddings_for_entities(entity_texts)

    entities_info = []
    for i, entity_detail in enumerate(valid_ner_entities_for_embedding):
        _all_sents, containing_sentence = get_entity_containing_sentences(text, entity_detail['text'])
        if containing_sentence: 
            entities_info.append({
                'text_span': entity_detail['text'],
                'type': entity_detail['label'].upper(), 
                'embedding': entity_embeddings[i], # Corresponds to entity_detail
                'sentence': containing_sentence,
                'original_label': entity_detail['label'] 
            })
    return entities_info

def compare_entities_for_edits(gen_entities_info, ref_entities_info, 
                              affinity_matrix, neg_class_list, neg_weight, threshold, device):
    """
    Compares generated and reference entities to find contradictions, FPs, and FNs.
    """
    contradictions = []
    false_positives = []
    false_negatives = []

    if not gen_entities_info and not ref_entities_info:
        return contradictions, false_positives, false_negatives
    if not ref_entities_info: 
        false_positives.extend(gen_entities_info)
        return contradictions, false_positives, false_negatives
    if not gen_entities_info: 
        false_negatives.extend(ref_entities_info)
        return contradictions, false_positives, false_negatives

    # Ensure embeddings are present before stacking
    valid_gen_entities_info = [e for e in gen_entities_info if 'embedding' in e and e['embedding'] is not None]
    valid_ref_entities_info = [e for e in ref_entities_info if 'embedding' in e and e['embedding'] is not None]

    if not valid_gen_entities_info or not valid_ref_entities_info:
        # Handle cases where one list might become empty after filtering
        if not valid_ref_entities_info and valid_gen_entities_info:
            false_positives.extend(valid_gen_entities_info)
        if not valid_gen_entities_info and valid_ref_entities_info:
            false_negatives.extend(valid_ref_entities_info)
        return contradictions, false_positives, false_negatives
        
    gen_embeddings = torch.stack([e['embedding'] for e in valid_gen_entities_info]).to(device)
    ref_embeddings = torch.stack([e['embedding'] for e in valid_ref_entities_info]).to(device)

    sim_matrix = cosine_similarity(gen_embeddings, ref_embeddings)

    gen_to_ref_best_match_idx = [-1] * len(valid_gen_entities_info)
    gen_to_ref_best_match_score = [-1.0] * len(valid_gen_entities_info)
    
    for i, gen_entity in enumerate(valid_gen_entities_info):
        best_j = -1
        max_effective_score = -float('inf')
        for j, ref_entity in enumerate(valid_ref_entities_info):
            cos_sim = sim_matrix[i, j].item()
            affinity_key = (gen_entity['type'], ref_entity['type'])
            affinity_w = affinity_matrix.get(affinity_key, 0.0) 
            type_penalty = neg_weight if (gen_entity['type'], ref_entity['type']) in neg_class_list or \
                                      (ref_entity['type'], gen_entity['type']) in neg_class_list else 1.0
            effective_score = cos_sim * affinity_w * type_penalty
            if effective_score > max_effective_score:
                max_effective_score = effective_score
                best_j = j

        if best_j != -1:
            gen_to_ref_best_match_idx[i] = best_j
            gen_to_ref_best_match_score[i] = max_effective_score
            best_ref_entity = valid_ref_entities_info[best_j]
            cos_sim_for_contradiction_check = sim_matrix[i, best_j].item()
            is_neg_pair = (gen_entity['type'], best_ref_entity['type']) in neg_class_list or \
                          (best_ref_entity['type'], gen_entity['type']) in neg_class_list
            if is_neg_pair and cos_sim_for_contradiction_check >= threshold : 
                contradictions.append({
                    'gen_entity': gen_entity, 'ref_entity': best_ref_entity,
                    'similarity': cos_sim_for_contradiction_check,
                    'effective_score_for_match_debug': max_effective_score
                })

    for i, gen_entity in enumerate(valid_gen_entities_info):
        if gen_to_ref_best_match_score[i] < threshold:
            false_positives.append(gen_entity)

    ref_to_gen_best_match_score = [-1.0] * len(valid_ref_entities_info)
    for j, ref_entity in enumerate(valid_ref_entities_info):
        best_i = -1
        max_effective_score = -float('inf')
        for i, gen_entity in enumerate(valid_gen_entities_info):
            cos_sim = sim_matrix[i, j].item() 
            affinity_key = (gen_entity['type'], ref_entity['type'])
            affinity_w = affinity_matrix.get(affinity_key, 0.0)
            type_penalty = neg_weight if (gen_entity['type'], ref_entity['type']) in neg_class_list or \
                                      (ref_entity['type'], gen_entity['type']) in neg_class_list else 1.0
            effective_score = cos_sim * affinity_w * type_penalty
            if effective_score > max_effective_score:
                max_effective_score = effective_score
                best_i = i
        if best_i != -1:
             ref_to_gen_best_match_score[j] = max_effective_score
    
    for j, ref_entity in enumerate(valid_ref_entities_info):
        if ref_to_gen_best_match_score[j] < threshold:
            is_part_of_contradiction = any(
                contr['ref_entity']['text_span'] == ref_entity['text_span'] and \
                contr['ref_entity']['sentence'] == ref_entity['sentence'] for contr in contradictions
            )
            if not is_part_of_contradiction:
                false_negatives.append(ref_entity)
                
    # For serialization, convert embeddings to lists
    for entity in contradictions + false_positives + false_negatives:
        if 'gen_entity' in entity and 'embedding' in entity['gen_entity']:
            if isinstance(entity['gen_entity']['embedding'], torch.Tensor):
                entity['gen_entity']['embedding'] = entity['gen_entity']['embedding'].cpu().tolist()
            elif not isinstance(entity['gen_entity']['embedding'], list):
                entity['gen_entity']['embedding'] = []
        if 'ref_entity' in entity and 'embedding' in entity['ref_entity']:
            if isinstance(entity['ref_entity']['embedding'], torch.Tensor):
                entity['ref_entity']['embedding'] = entity['ref_entity']['embedding'].cpu().tolist()
            elif not isinstance(entity['ref_entity']['embedding'], list):
                entity['ref_entity']['embedding'] = []
        if 'embedding' in entity:
            if isinstance(entity['embedding'], torch.Tensor):
                entity['embedding'] = entity['embedding'].cpu().tolist()
            elif not isinstance(entity['embedding'], list):
                entity['embedding'] = []
    
    return contradictions, false_positives, false_negatives

def analyze_text_pair(gen_text, gt_text, rs_models):
    """Process a pair of texts and return entity analysis results."""
    # Extract entities from texts
    gen_entities_info = extract_entities_with_context(gen_text, rs_models)
    ref_entities_info = extract_entities_with_context(gt_text, rs_models)
    
    # Compare entities to find contradictions, false positives, and false negatives
    contradictions, false_positives, false_negatives = compare_entities_for_edits(
        gen_entities_info, ref_entities_info, 
        PROCESSED_AFFINITY_MATRIX, DEFAULT_NEG_CLASS, DEFAULT_NEG_WEIGHT, 
        DEFAULT_SIMILARITY_THRESHOLD, rs_models.device
    )
    
    # Group entities by sentence for replacement operations
    ref_entities_info_grouped_by_sentence = {}
    for entity in ref_entities_info:
        if 'sentence' in entity:
            if entity['sentence'] not in ref_entities_info_grouped_by_sentence:
                ref_entities_info_grouped_by_sentence[entity['sentence']] = []
            ref_entities_info_grouped_by_sentence[entity['sentence']].append(entity)
    
    # Return analysis results
    return {
        "contradictions": contradictions,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "ref_entities_grouped_by_sentence": ref_entities_info_grouped_by_sentence
    }

# Initialize Flask app
app = Flask(__name__)

# Initialize model at application startup
logger.info("Initializing RaTEScore models...")
rs_models = RateScoreModels()
logger.info("RaTEScore models initialized successfully")

def wrap_message(status, data):
    return {"result": status, "data": data}

def wrap_error(message):
    return wrap_message("error", {"message": message})

def wrap_predictions(predictions):
    return wrap_message("success", {"predictions": predictions})

def convert_tensors_to_serializable(obj):
    """Recursively convert any torch.Tensor objects to Python lists for JSON serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu().detach().tolist()  # Handle CUDA tensors too
    elif isinstance(obj, dict):
        return {k: convert_tensors_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_tensors_to_serializable(v) for v in obj]
    else:
        return obj

@app.route('/analyze', methods=['POST'])
def analyze():
    logger.debug("Received request to /analyze endpoint")
    try:
        # Log request data
        data = request.json
        logger.debug(f"Request data: num_generations={len(data.get('generations', []))}, num_ground_truths={len(data.get('ground_truth', []))}")
        
        # Process the first example for debugging
        if data.get('generations') and len(data.get('generations')) > 0:
            logger.debug(f"Sample generation: {data['generations'][0][:100]}...")
        if data.get('ground_truth') and len(data.get('ground_truth')) > 0:
            logger.debug(f"Sample ground truth: {data['ground_truth'][0][:100]}...")
        
        if request.json is None:
            return json_response(wrap_error("Expected a JSON request"), 400)

        data = request.json
        generations = data.get("generations", [])
        ground_truth = data.get("ground_truth", [])
        
        if not generations:
            return json_response(wrap_error("No generations provided"), 400)
        
        # Ensure ground_truth is a list. If client sends a single string, wrap it in a list.
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth] 
        elif not isinstance(ground_truth, list):
            return json_response(wrap_error("Ground truth must be a string or a list of strings"), 400)

        # Validate lengths match
        if len(ground_truth) != len(generations):
            return json_response(wrap_error(f"Mismatch between generations ({len(generations)}) and ground truth ({len(ground_truth)})"), 400)

        current_app.logger.info(f"Received {len(generations)} generations for analysis.")
        
        try:
            # Process each generation and ground truth pair
            results = []
            for i, (gen_text, gt_text) in enumerate(zip(generations, ground_truth)):
                current_app.logger.info(f"Analyzing pair {i+1}/{len(generations)}")
                analysis = analyze_text_pair(gen_text, gt_text, rs_models)
                results.append(analysis)
            
            # Apply conversion to analysis_results before returning
            serializable_analysis_results = convert_tensors_to_serializable(results)
            
            return json_response(wrap_predictions({
                "analysis_results": serializable_analysis_results
            }))
            
        except Exception as e:
            # Log the full exception for better debugging on the server
            error_message = str(e)
            tb_str = traceback.format_exc()
            current_app.logger.error(f"Error processing request: {error_message}\nTraceback:\n{tb_str}")
            return json_response(wrap_error(error_message), 500)
    except Exception as e:
        logger.error(f"Error in /analyze endpoint: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    logger.info(f"Running on node: {os.uname().nodename}")
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5001) 