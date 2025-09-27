from flask import Flask, request
from flask_responses import json_response
import torch
from radgraph import F1RadGraph
from f1chexbert import WeightedCheXbert, F1CheXbert
import json
import re
import os
import argparse
from RaTEScore import RaTEScore
# import statistics # Commented out as per user request

# Initialize RadGraph
# f1radgraph = F1RadGraph(reward_level="partial", device="cuda")
f1radgraph = F1RadGraph(reward_level="all", device="cuda")
bi14chexbert = WeightedCheXbert(device="cuda")
f1chexbert = F1CheXbert(device="cuda")
ratescore = RaTEScore()

def extract_boxed_content(text: str) -> str:
    """
    Extracts answers in \\boxed{}.
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{") :]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1

            if depth == -1:  # exit
                end_pos = i
                break

    if end_pos != -1:
        return content[:end_pos].strip()

    return "None"

app = Flask(__name__)

def wrap_message(status, data):
    return {"result": status, "data": data}

def wrap_error(message):
    return wrap_message("error", {"message": message})

def wrap_predictions(predictions):
    return wrap_message("success", {"predictions": predictions})

# def r1v_format_reward(predict_str: str) -> float:
#     pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
#     format_match = re.fullmatch(pattern, predict_str)
#     return 1.0 if format_match else 0.0

def easyr1_math_format_reward(predict_str: str, type='short') -> float:
    # pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL) # potential issue for zero format reward, model output '\box' instead of '\\box..'
    # pattern = re.compile(r"<think>.*</think>.*\boxed\{.*\}.*", re.DOTALL)
    
    if type == 'short':
        # for short prompt wo/ cot
        pattern = re.compile(r"\\boxed\{.*\}.*", re.DOTALL)
        format_match = re.fullmatch(pattern, predict_str)
    elif type == 'cot':
        pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
        format_match = re.fullmatch(pattern, predict_str)
    elif type == 'free_form':
        return 1.0
    else:
        print("no implementation")
        return 0.0
    # Pattern that accepts both \\boxed and \boxed
    # pattern = re.compile(r"\\?\\boxed\{[^{}]*\}", re.DOTALL)
    # format_match = re.search(pattern, predict_str)

    return 1.0 if format_match else 0.0

def load_class_frequencies(file_path):
    """
    Load class_frequencies from a JSON file.
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as json_file:
            return json.load(json_file)
    else:
        raise FileNotFoundError(f"File {file_path} not found.")

@app.route("/predict", methods=["POST"])
def predict():
    if request.json is None:
        return json_response(wrap_error("Expected a JSON request"), 400)

    if "hyps" not in request.json or "refs" not in request.json:
        return json_response(wrap_error("Missing 'hyps' or 'refs' in request"), 400)

    hyps = request.json["hyps"]
    refs = request.json["refs"]
    type = request.json.get("type", "short")  # Default to 'short' if not provided
    requested_reward_components = request.json.get("requested_reward_components", []) # Added: Get requested components

    # class_frequencies_path = request.json.get("class_frequencies_path", "class_frequencies.json")  # Default path if not provided

    # check the outputs of the model
    # print("predictions:")
    # print(hyps)
    # print("references:")
    # print(refs)
    # print("type:")
    # print(type)

    if not isinstance(hyps, list) or not isinstance(refs, list):
        return json_response(wrap_error("'hyps' and 'refs' should be lists"), 400)

    try:
        # # Load class_frequencies from the provided path
        # class_frequencies = load_class_frequencies(class_frequencies_path)

        # Compute RadGraph score - convert single score to list
        if type != 'free_form':
            processed_hyps = [extract_boxed_content(h) for h in hyps]  # Process ALL hyps, not just hyps[0]
            score_radgraph, _, _, _ = f1radgraph(hyps=processed_hyps, refs=refs)
            score_ratescore = ratescore.compute_score(processed_hyps, refs)
        else:
            score_radgraph, _, _, _ = f1radgraph(hyps=hyps, refs=refs)
            score_ratescore = ratescore.compute_score(hyps, refs)
        score_radgraph = score_radgraph[1] # extract partial rewards (mean_partial_precision, mean_partial_recall, mean_partial_f1)
        prec_radgraph = [score_radgraph[0]] if not isinstance(score_radgraph[0], list) else score_radgraph[0]
        recall_radgraph = [score_radgraph[1]] if not isinstance(score_radgraph[1], list) else score_radgraph[1]
        f1_radgraph = [score_radgraph[2]] if not isinstance(score_radgraph[2], list) else score_radgraph[2]
        
        # Compute RaTEScore
        # Take only the mean score (first element)
        score_ratescore = score_ratescore[0]  # Get the mean score, ignore std

        # Ensure score_ratescore is a list for consistent processing
        if not isinstance(score_ratescore, list):
            score_ratescore = [score_ratescore]
        
        # ratescore_values will be the list of scores, e.g., [score_value] or [s1, s2]
        # The key in the output json will be "rate_scores"
        output_rate_scores = score_ratescore if score_ratescore and all(isinstance(x, (int, float)) for x in score_ratescore) else [0.0]

        # Compute ChexBERT score - convert single score to list
        if type != 'free_form':
            processed_hyps = [extract_boxed_content(h) for h in hyps]  # Process ALL hyps, not just hyps[0]
            cxb_accuracy, _, class_report_14, class_report_5 = f1chexbert(hyps=processed_hyps, refs=refs)
        else:
            cxb_accuracy, _, class_report_14, class_report_5 = f1chexbert(hyps=hyps, refs=refs)
        cxb_14_micro = class_report_14["micro avg"]["f1-score"]
        cxb_14_macro = class_report_14["macro avg"]["f1-score"]
        cxb_5_micro = class_report_5["micro avg"]["f1-score"]
        cxb_5_macro = class_report_5["macro avg"]["f1-score"]
        # convert to the list format for flask server prediction
        cxb_accuracy = [cxb_accuracy] if not isinstance(cxb_accuracy, list) else cxb_accuracy
        cxb_14_micro = [cxb_14_micro] if not isinstance(cxb_14_micro, list) else cxb_14_micro
        cxb_14_macro = [cxb_14_macro] if not isinstance(cxb_14_macro, list) else cxb_14_macro
        cxb_5_micro = [cxb_5_micro] if not isinstance(cxb_5_micro, list) else cxb_5_micro
        cxb_5_macro = [cxb_5_macro] if not isinstance(cxb_5_macro, list) else cxb_5_macro

        # Compute ChexBERT binary accuracy score - convert single score to list
        if type != 'free_form':
            processed_hyps = [extract_boxed_content(h) for h in hyps]  # Process ALL hyps, not just hyps[0]
            cxb_14_results = bi14chexbert(hyps=processed_hyps, refs=refs, class_frequencies=class_frequencies)
        else:
            cxb_14_results = bi14chexbert(hyps=hyps, refs=refs, class_frequencies=class_frequencies)
        cxb_14_overall_weighted_acc = cxb_14_results["overall"]["weighted_accuracy"]
        # convert to the list format for flask server prediction
        cxb_14_overall_weighted_acc = [cxb_14_overall_weighted_acc] if not isinstance(cxb_14_overall_weighted_acc, list) else cxb_14_overall_weighted_acc
        
        # Compute format scores with the specified type
        format_scores = [easyr1_math_format_reward(h, type=type) for h in hyps]
        
        # Store all computed scores in a dictionary to easily access them by name
        all_computed_metrics = {
            "f1_radgraph": f1_radgraph[0] if f1_radgraph else 0.0, # Assuming f1_radgraph is a list with one item
            "precision_radgraph": prec_radgraph[0] if prec_radgraph else 0.0,
            "recall_radgraph": recall_radgraph[0] if recall_radgraph else 0.0,
            "balanced_accuracy_chexbert": cxb_14_overall_weighted_acc[0] if cxb_14_overall_weighted_acc else 0.0,
            "format_scores": format_scores[0] if format_scores else 0.0, # Assuming format_scores is a list with one item
            "accuracy_chexbert": cxb_accuracy[0] if cxb_accuracy else 0.0,
            "micro_chexbert_14": cxb_14_micro[0] if cxb_14_micro else 0.0,
            "macro_chexbert_14": cxb_14_macro[0] if cxb_14_macro else 0.0,
            "micro_chexbert_5": cxb_5_micro[0] if cxb_5_micro else 0.0,
            "macro_chexbert_5": cxb_5_macro[0] if cxb_5_macro else 0.0,
            "rate_score": output_rate_scores[0] if output_rate_scores else 0.0 # Use singular for internal calculation
            # Note: _scores often implies a list, but here we're taking the first element for calculation,
            # as the final score is usually singular per hyp-ref pair.
            # The lists are for batch consistency in flask response.
        }

        COMPONENT_MAPPING = { # Defined mapping
            'radgraphf1': 'f1_radgraph',
            'cxb14micro': 'micro_chexbert_14',
            'format': 'format_scores',
            'cxb_bal_acc': 'balanced_accuracy_chexbert',
            'ratescore': 'rate_score'
        }

        final_scores_val = 0
        if requested_reward_components == ["all"]:
            final_scores_val += all_computed_metrics.get("f1_radgraph", 0.0)
            final_scores_val += all_computed_metrics.get("balanced_accuracy_chexbert", 0.0) / 14.0
            final_scores_val += all_computed_metrics.get("rate_score", 0.0)
            final_scores_val += all_computed_metrics.get("micro_chexbert_14", 0.0)
            if type != 'free_form':
                final_scores_val += all_computed_metrics.get("format_scores", 0.0)
            final_scores = [final_scores_val]
        elif requested_reward_components: # Not ["all"] but a non-empty list of specific components
            for user_comp_name in requested_reward_components:
                internal_metric_key = COMPONENT_MAPPING.get(user_comp_name)
                if internal_metric_key:
                    value = all_computed_metrics.get(internal_metric_key, 0.0)
                    if user_comp_name == 'cxb_bal_acc':
                        final_scores_val += value / 14.0 # Apply scaling
                    elif user_comp_name == 'format':
                        if type != 'free_form': # Conditional addition
                            final_scores_val += value
                    else:
                        final_scores_val += value # Direct addition for others
            final_scores = [final_scores_val]
        else: # requested_reward_components is empty (e.g., not provided by client)
            # Default final score calculation (user's updated logic)
            rad_component = all_computed_metrics.get("f1_radgraph", 0.0)
            cxb_bal_acc_component = all_computed_metrics.get("balanced_accuracy_chexbert", 0.0) / 14.0
            fmt_component = all_computed_metrics.get("format_scores", 0.0)
            rate_component = all_computed_metrics.get("rate_score", 0.0)
            cxb_micro_component = all_computed_metrics.get("micro_chexbert_14", 0.0)
            if type == 'free_form':
                final_scores = [rad_component + cxb_bal_acc_component + rate_component + cxb_micro_component]
            else:
                final_scores = [rad_component + cxb_bal_acc_component + fmt_component + rate_component + cxb_micro_component]
        
        return json_response(wrap_predictions({
            'final_scores': final_scores,
            'precision_radgraph_scores': prec_radgraph,
            'recall_radgraph_scores': recall_radgraph,
            'f1_radgraph_scores': f1_radgraph,
            'balanced_accuracy_chexbert': cxb_14_overall_weighted_acc,
            'format_scores': format_scores,
            'accuracy_chexbert': cxb_accuracy,
            'micro_chexbert_14': cxb_14_micro,
            'macro_chexbert_14': cxb_14_macro,
            'micro_chexbert_5': cxb_5_micro,
            'macro_chexbert_5': cxb_5_macro,
            'rate_scores': output_rate_scores, # Return the list of rate_scores 
        }))
    except Exception as e:
        return json_response(wrap_error(str(e)), 500)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--class-frequencies-path', default='./class_frequencies_iu_xray_train.json')
    args = parser.parse_args()
    
    # Store path in a global variable or pass it to necessary functions
    class_frequencies = load_class_frequencies(args.class_frequencies_path)
    print("successfully loaded class frequencies: {}".format(args.class_frequencies_path))
    app.run(host="0.0.0.0", port=5000) 