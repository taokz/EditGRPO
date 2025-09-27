from flask import Flask, request
from flask_responses import json_response
import torch
import json
import re
import os
import argparse
from green_score import GREEN

# Initialize GREEN scorer
# Default configuration - can be overridden via initialization parameters
MODEL_NAME = "StanfordAIMI/GREEN-radllama2-7b"
# MODEL_NAME = "PrunaAI/StanfordAIMI-GREEN-RadLlama2-7b-bnb-4bit-smashed"
OUTPUT_DIR = "."
USE_VLLM = False  # Can be set to True if vLLM is available
BATCH_SIZE = 16

# Global variable to store the GREEN scorer instance
green_scorer = None

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

def easyr1_math_format_reward(predict_str: str, type='short') -> float:
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

    return 1.0 if format_match else 0.0

def initialize_green_scorer():
    """Initialize the GREEN scorer with specified configuration"""
    global green_scorer
    
    if green_scorer is None:
        print(f"Initializing GREEN scorer with model: {MODEL_NAME}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Use vLLM: {USE_VLLM}")
        print(f"Batch size: {BATCH_SIZE}")
        
        try:
            green_scorer = GREEN(
                model_name=MODEL_NAME,
                output_dir=OUTPUT_DIR,
                use_vllm=USE_VLLM,
                batch_size=BATCH_SIZE
            )
            print(f"GREEN scorer initialized successfully with backend: {'vLLM' if green_scorer.use_vllm else 'HuggingFace'}")
        except Exception as e:
            print(f"Error initializing GREEN scorer: {str(e)}")
            raise e
    
    return green_scorer

@app.route("/predict", methods=["POST"])
def predict():
    if request.json is None:
        return json_response(wrap_error("Expected a JSON request"), 400)

    if "hyps" not in request.json or "refs" not in request.json:
        return json_response(wrap_error("Missing 'hyps' or 'refs' in request"), 400)

    hyps = request.json["hyps"]
    refs = request.json["refs"]
    type = request.json.get("type", "short")  # Default to 'short' if not provided
    requested_reward_components = request.json.get("requested_reward_components", [])  # Get requested components

    if not isinstance(hyps, list) or not isinstance(refs, list):
        return json_response(wrap_error("'hyps' and 'refs' should be lists"), 400)

    try:
        # Initialize GREEN scorer if not already done
        scorer = initialize_green_scorer()
        
        # Prepare hypotheses for GREEN scoring based on type
        if type != 'free_form':
            processed_hyps = [extract_boxed_content(h) for h in hyps]
        else:
            processed_hyps = hyps
        
        # Compute GREEN score
        # GREEN scorer returns: mean, std, green_score_list, summary, result_df
        mean_green, std_green, green_score_list, summary, result_df = scorer(processed_hyps, refs)
        
        # Convert to list format for consistency with other flask servers
        mean_green_list = [mean_green] if not isinstance(mean_green, list) else mean_green
        std_green_list = [std_green] if not isinstance(std_green, list) else std_green
        
        # Compute format scores with the specified type
        format_scores = [easyr1_math_format_reward(h, type=type) for h in hyps]
        
        # Store all computed metrics in a dictionary to easily access them by name
        all_computed_metrics = {
            "green_score": mean_green,
            "green_mean": mean_green,
            "green_std": std_green,
            "format_scores": format_scores[0] if format_scores else 0.0,
        }

        # Component mapping for GREEN-specific components
        COMPONENT_MAPPING = {
            'green': 'green_score',
            'format': 'format_scores',
        }

        final_scores_val = 0
        if requested_reward_components == ["all"]:
            # Include all available GREEN components
            final_scores_val += all_computed_metrics.get("green_score", 0.0)
            if type != 'free_form':
                final_scores_val += all_computed_metrics.get("format_scores", 0.0)
            final_scores = [final_scores_val]
        elif requested_reward_components:  # Not ["all"] but a non-empty list of specific components
            for user_comp_name in requested_reward_components:
                internal_metric_key = COMPONENT_MAPPING.get(user_comp_name)
                if internal_metric_key:
                    value = all_computed_metrics.get(internal_metric_key, 0.0)
                    if user_comp_name == 'format':
                        if type != 'free_form':  # Conditional addition
                            final_scores_val += value
                    else:
                        final_scores_val += value  # Direct addition for others
            final_scores = [final_scores_val]
        else:  # requested_reward_components is empty (e.g., not provided by client)
            # Default final score calculation
            green_component = all_computed_metrics.get("green_score", 0.0)
            fmt_component = all_computed_metrics.get("format_scores", 0.0)
            if type == 'free_form':
                final_scores = [green_component]
            else:
                final_scores = [green_component + fmt_component]
        
        return json_response(wrap_predictions({
            'final_scores': final_scores,
            'green_mean': mean_green_list,
            'green_std': std_green_list,
            'format_scores': format_scores,
            'summary': summary if summary else {},
        }))
    except Exception as e:
        return json_response(wrap_error(str(e)), 500)

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    try:
        scorer = initialize_green_scorer()
        return json_response(wrap_message("success", {
            "status": "healthy",
            "model_name": MODEL_NAME,
            "use_vllm": scorer.use_vllm if scorer else USE_VLLM,
            "batch_size": BATCH_SIZE
        }))
    except Exception as e:
        return json_response(wrap_error(f"Health check failed: {str(e)}"), 500)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default=MODEL_NAME, help='GREEN model name to use')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='Output directory for GREEN')
    parser.add_argument('--use-vllm', action='store_true', help='Use vLLM backend instead of HuggingFace')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size for GREEN scorer')
    parser.add_argument('--port', type=int, default=5002, help='Port to run the server on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to run the server on')
    args = parser.parse_args()
    
    # Update global configuration
    MODEL_NAME = args.model_name
    OUTPUT_DIR = args.output_dir
    USE_VLLM = args.use_vllm
    BATCH_SIZE = args.batch_size
    
    print(f"Starting GREEN Flask server with configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Use vLLM: {USE_VLLM}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    
    # Initialize GREEN scorer on startup
    try:
        initialize_green_scorer()
        print("GREEN scorer pre-initialized successfully")
    except Exception as e:
        print(f"Warning: Could not pre-initialize GREEN scorer: {str(e)}")
        print("Will attempt to initialize on first request")
    
    app.run(host=args.host, port=args.port) 


# python green_flask_server.py --use-vllm --batch-size 32 --port 5001