import re
from typing import Dict

import nltk
from nltk.translate.bleu_score import sentence_bleu

def bleu4_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0

def bleu4_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    try:
        # Tokenize the strings
        reference = [ground_truth.split()]
        candidate = predict_str.split()
        
        # Calculate BLEU-4 score
        score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
        return score
    except Exception:
        return 0.0

def bleu4_compute_score(predict_str: str, ground_truth: str) -> Dict[str, float]:
    format = bleu4_format_reward(predict_str)
    accuracy = bleu4_accuracy_reward(predict_str, ground_truth)
    return {
        "overall": 0.9 * accuracy + 0.1 * format,
        "format": format,
        "accuracy": accuracy,
    }
