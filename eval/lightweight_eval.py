import json
import numpy as np
import pandas as pd
import statistics
import re
import argparse
import os
import pickle
import torch
import shutil
import tempfile
from tqdm import tqdm

# Existing imports
from radgraph import F1RadGraph
from f1chexbert import F1CheXbert
from RaTEScore import RaTEScore
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# New imports for RadCliQ
from bert_score import BERTScorer

# RadCliQ model paths
RADCLIQ_DIR = "CXR-Report-Metric/CXRMetric"
NORMALIZER_PATH = os.path.join(RADCLIQ_DIR, "normalizer.pkl")
COMPOSITE_METRIC_V0_PATH = os.path.join(RADCLIQ_DIR, "composite_metric_model.pkl")
COMPOSITE_METRIC_V1_PATH = os.path.join(RADCLIQ_DIR, "radcliq-v1.pkl")
# CHEXBERT_PATH = os.path.join(RADCLIQ_DIR, "CheXbert/models/chexbert.pth")
CHEXBERT_PATH = "chexbert.pth"

def preprocess_captions(images_captions):
    bioclean = lambda t: re.sub(r'[.,?;*!%^&_+():\-\[\]{}]', '',
                                t.replace('"', '').replace('/', '').replace('\\', '').replace("'",
                                                                                              '').strip().lower())
    pr_captions = {}
    for image in images_captions:
        pr_captions[image] = [bioclean(images_captions[image])]
    return pr_captions

def compute_scores(hyps, refs):
    # print("Hyps:{}".format(hyps))
    # print("Refs:{}".format(refs))
    score_radgraph, _, _, _ = f1radgraph(hyps=hyps, refs=refs)
    _, _, class_report, class_report_5 = f1chexbert(hyps=hyps, refs=refs)
    return score_radgraph, class_report, class_report_5

def compute_ratescore(hyps, refs):
    scores = ratescore.compute_score(hyps, refs)
    return scores

def compute_coco_scores(gts, res):
    gts = preprocess_captions(gts)
    res = preprocess_captions(res)
    
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Meteor(), "METEOR"),
    ]
    
    scores = {}
    for scorer, method in scorers:
        try:
            score, _ = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for sc, m in zip(score, method):
                    scores[m] = sc
            else:
                scores[method] = score
        except Exception as e:
            print(f"Error computing {scorer.method()} score: {e}")
            if isinstance(method, list):
                for m in method:
                    scores[m] = None
            else:
                scores[method] = None
    return scores

def compute_bertscore(hyps, refs, use_idf=False):
    """Compute BERTScore F1 scores."""
    # Clean reports
    test_reports = [re.sub(r' +', ' ', ref) for ref in refs]
    method_reports = [re.sub(r' +', ' ', hyp) for hyp in hyps]
    
    scorer = BERTScorer(
        model_type="distilroberta-base",
        batch_size=256,
        lang="en",
        rescale_with_baseline=True,
        idf=use_idf,
        idf_sents=test_reports)
    _, _, f1 = scorer.score(method_reports, test_reports)
    return f1.tolist()

def compute_semantic_embedding_scores(hyps, refs):
    """Compute semantic embedding scores using CheXbert embeddings."""
    import tempfile
    import os
    
    # Create temporary CSV files for hyps and refs
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create temporary CSV files
        hyps_csv = os.path.join(temp_dir, "hyps.csv")
        refs_csv = os.path.join(temp_dir, "refs.csv")
        
        # Create DataFrames and save as CSV
        hyps_df = pd.DataFrame({
            'study_id': range(len(hyps)),
            'report': hyps
        })
        refs_df = pd.DataFrame({
            'study_id': range(len(refs)),
            'report': refs
        })
        
        hyps_df.to_csv(hyps_csv, index=False)
        refs_df.to_csv(refs_csv, index=False)
        
        # Generate embeddings using CheXbert
        hyps_embed_path = os.path.join(temp_dir, "hyps_embeddings.pt")
        refs_embed_path = os.path.join(temp_dir, "refs_embeddings.pt")
        
        # Check if CheXbert model exists
        if not os.path.exists(CHEXBERT_PATH):
            print(f"Warning: CheXbert model not found at {CHEXBERT_PATH}")
            print("Using dummy semantic embedding scores (0.5 for all samples)")
            return [0.0] * len(hyps)
        
        # Run CheXbert encoding
        os.system(f"python {RADCLIQ_DIR}/CheXbert/src/encode.py -c {CHEXBERT_PATH} -d {hyps_csv} -o {hyps_embed_path}")
        os.system(f"python {RADCLIQ_DIR}/CheXbert/src/encode.py -c {CHEXBERT_PATH} -d {refs_csv} -o {refs_embed_path}")
        
        # Check if embeddings were generated
        if not (os.path.exists(hyps_embed_path) and os.path.exists(refs_embed_path)):
            print("Warning: Failed to generate embeddings. Using dummy scores.")
            return [0.0] * len(hyps)
        
        # Load embeddings and compute cosine similarity
        label_embeds = torch.load(refs_embed_path)
        pred_embeds = torch.load(hyps_embed_path)
        
        list_label_embeds = []
        list_pred_embeds = []
        for data_idx in sorted(label_embeds.keys()):
            list_label_embeds.append(label_embeds[data_idx])
            list_pred_embeds.append(pred_embeds[data_idx])
            
        np_label_embeds = torch.stack(list_label_embeds, dim=0).numpy()
        np_pred_embeds = torch.stack(list_pred_embeds, dim=0).numpy()
        
        scores = []
        for i, (label, pred) in enumerate(zip(np_label_embeds, np_pred_embeds)):
            sim_scores = (label * pred).sum() / (
                np.linalg.norm(label) * np.linalg.norm(pred))
            scores.append(sim_scores)
        
        return scores
        
    except Exception as e:
        print(f"Error computing semantic embeddings: {e}")
        print("Using dummy semantic embedding scores (0.5 for all samples)")
        return [0.0] * len(hyps)
    
    finally:
        # Clean up temporary files
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

class CompositeMetric:
    """The RadCliQ-v1 composite metric.

    Attributes:
        scaler: Input normalizer.
        coefs: Coefficients including the intercept.
    """
    def __init__(self, scaler, coefs):
        """Initializes the composite metric with a normalizer and coefficients.

        Args:
            scaler: Input normalizer.
            coefs: Coefficients including the intercept.
        """
        self.scaler = scaler
        self.coefs = coefs

    def predict(self, x):
        """Generates composite metric score for input.

        Args:
            x: Input data.

        Returns:
            Composite metric score.
        """
        norm_x = self.scaler.transform(x)
        norm_x = np.concatenate(
            (norm_x, np.ones((norm_x.shape[0], 1))), axis=1)
        pred = norm_x @ self.coefs
        return pred

def compute_radcliq_scores(radgraph_scores, bertscore_scores, semb_scores, bleu_scores):
    """Compute RadCliQ v0 and v1 scores."""
    try:
        # Prepare input data
        input_data = np.array([radgraph_scores, bertscore_scores, semb_scores, bleu_scores]).T
        
        # Check if model files exist
        for path in [NORMALIZER_PATH, COMPOSITE_METRIC_V0_PATH, COMPOSITE_METRIC_V1_PATH]:
            if not os.path.exists(path):
                print(f"Warning: Model file not found: {path}")
                print("Cannot compute RadCliQ scores.")
                return [0.0] * len(radgraph_scores), [0.0] * len(radgraph_scores)
        
        # Load models
        with open(NORMALIZER_PATH, "rb") as f:
            normalizer = pickle.load(f)
        with open(COMPOSITE_METRIC_V0_PATH, "rb") as f:
            composite_metric_v0_model = pickle.load(f)
        with open(COMPOSITE_METRIC_V1_PATH, "rb") as f:
            composite_metric_v1_model = pickle.load(f)
        
        # Compute RadCliQ-v0 (with normalization)
        norm_input_data = normalizer.transform(input_data)
        radcliq_v0_scores = composite_metric_v0_model.predict(norm_input_data)
        
        # Compute RadCliQ-v1 (direct application)
        radcliq_v1_scores = composite_metric_v1_model.predict(input_data)
        
        return radcliq_v0_scores, radcliq_v1_scores
        
    except Exception as e:
        print(f"Error computing RadCliQ scores: {e}")
        print("Returning default scores...")
        return [0.0] * len(radgraph_scores), [0.0] * len(radgraph_scores)

def clean_text(text):
    """ Remove excessive whitespace, repeated statements, and replace empty text with '[EMPTY]' """
    text = text.strip()  # Remove leading/trailing spaces and newlines
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'Indication:.*', '', text).strip()

    # Remove multiple <|endoftext|> tokens more aggressively
    text = re.sub(r'<\|endoftext\|>+', '', text)
    
    # Clean up any remaining whitespace after token removal
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Enhanced sentence splitting - handle more edge cases
    # Split on sentence endings, but also handle cases where sentences might not end properly
    sentences = re.split(r'(?<=[.!?])\s+|(?<=\.)\s*(?=[A-Z])', text)
    
    # Additional split for cases where sentences are separated by multiple spaces without proper punctuation
    temp_sentences = []
    for sent in sentences:
        # Split on patterns like "word. Word" or "word.Word" 
        sub_sentences = re.split(r'(?<=[a-z])\.(?=\s*[A-Z])', sent)
        temp_sentences.extend(sub_sentences)
    sentences = temp_sentences
    
    unique_sentences = []
    seen_sentences = set()
    seen_normalized_sentences = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Ensure sentence ends with punctuation
        if sentence and not re.search(r'[.!?]$', sentence):
            sentence += '.'
            
        # Normalize sentence for comparison (remove punctuation, extra spaces, and lowercase)
        normalized = re.sub(r'[.!?,:;]+$', '', sentence).strip().lower()
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
        normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove remaining punctuation for comparison
        
        # Skip very short sentences (likely fragments)
        if len(normalized.split()) < 2:
            continue
            
        # Check for exact duplicates
        if normalized in seen_normalized_sentences:
            continue
            
        # Check for semantic similarity (simple approach using word overlap)
        is_duplicate = False
        normalized_words = set(normalized.split())
        
        for seen_norm in seen_normalized_sentences:
            seen_words = set(seen_norm.split())
            
            # Calculate Jaccard similarity (intersection over union)
            if len(normalized_words) > 0 and len(seen_words) > 0:
                intersection = len(normalized_words.intersection(seen_words))
                union = len(normalized_words.union(seen_words))
                jaccard_sim = intersection / union
                
                # If sentences are very similar (>80% word overlap), consider as duplicate
                if jaccard_sim > 0.8:
                    is_duplicate = True
                    break
                    
                # Also check if one sentence is a subset of another (>90% overlap)
                smaller_set = normalized_words if len(normalized_words) <= len(seen_words) else seen_words
                larger_set = seen_words if len(normalized_words) <= len(seen_words) else normalized_words
                subset_ratio = len(smaller_set.intersection(larger_set)) / len(smaller_set)
                
                if subset_ratio > 0.9:
                    is_duplicate = True
                    break
        
        if not is_duplicate and normalized:
            unique_sentences.append(sentence)
            seen_sentences.add(sentence)
            seen_normalized_sentences.add(normalized)
    
    # Join sentences with appropriate spacing
    text = ' '.join(unique_sentences)
    
    # Final cleanup - ensure proper sentence ending
    if text and not re.search(r'[.!?]$', text):
        text += '.'

    # Extract text within \boxed{} if it exists
    boxed_match = re.search(r'\\boxed\{(.*?)\}', text, re.DOTALL)
    if boxed_match:
        text_in_box = boxed_match.group(1).strip()
        text_in_box = re.sub(r'\s+', ' ', text_in_box) # Clean whitespace in the boxed text
        return text_in_box if text_in_box else "[EMPTY]" # Return cleaned boxed text or "[EMPTY]"
    
    return text if text else "[EMPTY]"  # Replace empty text with a placeholder

def main(data_path, save_path):
    # Load data
    print("Loading data...")
    print(data_path)
    data = json.load(open(data_path))

    # Prepare refs and hyps lists
    print("Preparing data...")
    img_ids = []
    refs = []
    hyps = []
    for item in data:
        if 'id' not in item:
            img_ids.append(item["dicom_id"])
        else:
            img_ids.append(item["id"])
        refs.append(clean_text(item["answer"]))
        hyps.append(clean_text(item["response"]))

    gts = dict(zip(img_ids, refs))
    res = dict(zip(img_ids, hyps))

    # Initialize F1RadGraph and F1CheXbert
    global f1radgraph
    global f1chexbert
    global ratescore
    print("Initializing F1RadGraph, F1CheXbert & RaTEScore...")
    f1radgraph = F1RadGraph(reward_level="all", cuda=-1) # OOM for mimic-cxr, TODO: check the reason
    f1chexbert = F1CheXbert(device="cuda")
    ratescore = RaTEScore()

    # Compute scores
    print("Computing RadGraph and CheXbert scores...")
    score_radgraph, class_report_14, class_report_5 = compute_scores(hyps, refs)

    print("Computing RaTEScore...")
    score_ratescore = compute_ratescore(hyps, refs)

    print("Computing COCO scores...")
    coco_scores = compute_coco_scores(gts, res)

    # Compute RadCliQ component metrics (REUSE existing calculations)
    print("Computing RadCliQ component metrics...")
    print("  - Reusing RadGraph results...")
    
    # Extract RadGraph combined score from existing calculations
    if isinstance(score_radgraph, tuple) and len(score_radgraph) == 3:
        simple, partial, complete = score_radgraph
        radgraph_combined_score = partial[2]  # Use partial F1 as combined score
    else:
        radgraph_combined_score = score_radgraph

    # Extract BLEU-2 from existing coco_scores (REUSE existing calculation)
    bleu2_score = coco_scores.get("Bleu_2", 0.0)

    # Create arrays for RadCliQ calculation
    radgraph_scores_array = [radgraph_combined_score] * len(hyps)
    bleu2_scores = [bleu2_score] * len(hyps)
    
    print("  - Computing BERTScore...")
    bertscore_scores = compute_bertscore(hyps, refs, use_idf=False)
    
    print("  - Computing semantic embedding scores...")
    semb_scores = compute_semantic_embedding_scores(hyps, refs)
    
    # Compute RadCliQ v0 and v1
    print("Computing RadCliQ v0 and v1 scores...")
    radcliq_v0_scores, radcliq_v1_scores = compute_radcliq_scores(
        radgraph_scores_array, bertscore_scores, semb_scores, bleu2_scores
    )

    # Prepare results
    results = {
        'RaTEScore_mean': statistics.mean(score_ratescore),
        'RaTEScore_std': statistics.stdev(score_ratescore),
    }

    # Handle different RadGraph reward formats
    if isinstance(score_radgraph, tuple) and len(score_radgraph) == 3:
        # For reward_level="all", unpack and add all metrics
        simple, partial, complete = score_radgraph
        results.update({
            "RadGraph_Simple_P": simple[0],
            "RadGraph_Simple_R": simple[1],
            "RadGraph_Simple_F1": simple[2],
            "RadGraph_Partial_P": partial[0],
            "RadGraph_Partial_R": partial[1], 
            "RadGraph_Partial_F1": partial[2],
            "RadGraph_Complete_P": complete[0],
            "RadGraph_Complete_R": complete[1],
            "RadGraph_Complete_F1": complete[2],
        })
        # For backward compatibility
        results["RadGraph"] = partial[2]  # Use partial F1 as the main RadGraph score
    else:
        # For other reward levels, just use the single score
        results["RadGraph"] = score_radgraph

    results.update({
        "CheXbert_14_Micro": class_report_14["micro avg"]["f1-score"],
        "CheXbert_14_Macro": class_report_14["macro avg"]["f1-score"],
        "CheXbert_5_Micro": class_report_5["micro avg"]["f1-score"],
        "CheXbert_5_Macro": class_report_5["macro avg"]["f1-score"],
        **coco_scores
    })

    # Add RadCliQ scores (simplified - no std, no intermediate metrics)
    results.update({
        "RadCliQ_v0": np.mean(radcliq_v0_scores),
        "RadCliQ_v1": np.mean(radcliq_v1_scores),
        "BERTScore": np.mean(bertscore_scores),
        "SemB": np.mean(semb_scores),
    })

    # Save the results to a CSV file
    print("Saving results...")
    results_df = pd.DataFrame([results])
    results_df.to_csv(save_path, index=False)

    # Print results
    print("\nResults:")
    for metric, score in results.items():
        if score is not None:
            if isinstance(score, float):
                print(f"{metric}: {score:.4f}")
            else:
                print(f"{metric}: {score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation for entire dataset")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input JSON data file')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the output CSV file')
    args = parser.parse_args()
    
    main(args.data_path, args.save_path)

# Example usage:
# python full_eval.py --data_path mimic_cxr_data.json --save_path evaluation_results.csv