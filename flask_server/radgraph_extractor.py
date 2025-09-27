from radgraph import RadGraph
from flask import Flask, request, current_app # Added current_app for logger potentially
from flask_responses import json_response
import json
import traceback # For detailed error logging

# Initialize RadGraph
radgraph = RadGraph()

app = Flask(__name__)

def wrap_message(status, data):
    return {"result": status, "data": data}

def wrap_error(message):
    return wrap_message("error", {"message": message})

def wrap_predictions(predictions):
    return wrap_message("success", {"predictions": predictions})

@app.route('/extract', methods=['POST'])
def extract():
    """
    Extract annotations from generated text and ground truth.
    
    Expected JSON input:
    {
        "generations": ["text1", "text2", ...],
        "ground_truth": ["reference text1", "reference text2", ...]
    }
    
    Returns:
    {
        "result": "success",
        "data": {
            "predictions": {
                 "gen_annotations": {
                     "0": {"text": "...", "entities": {...}, ...},
                     "1": {...}
                 },
                 "gt_annotations": {
                     "0": {"text": "...", "entities": {...}, ...},
                     "1": {...}
                 }
            }
        }
    }
    """
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

    current_app.logger.info(f"Received {len(generations)} generations.")
    if generations:
        current_app.logger.info(f"Sample generation: '{generations[0][:100]}...'" if generations[0] else "Empty generation string")
    
    current_app.logger.info(f"Received {len(ground_truth)} ground truth texts.")
    if ground_truth:
         current_app.logger.info(f"Sample ground truth: '{ground_truth[0][:100]}...'" if ground_truth[0] else "Empty ground_truth string")
    
    try:
        gen_annotations_map = radgraph(generations)
        
        gt_annotations_map = {}
        if ground_truth: # Only process if ground_truth list is not empty
            gt_annotations_map = radgraph(ground_truth) # ground_truth is already a list of strings
        else:
            current_app.logger.info("No ground truth texts provided by client, gt_annotations_map will be empty.")

        return json_response(wrap_predictions({
            "gen_annotations": gen_annotations_map,
            "gt_annotations": gt_annotations_map # Return the map of GT annotations
        }))
        
    except Exception as e:
        # Log the full exception for better debugging on the server
        error_message = str(e)
        tb_str = traceback.format_exc()
        current_app.logger.error(f"Error processing request: {error_message}\nTraceback:\n{tb_str}")
        return json_response(wrap_error(error_message), 500)

if __name__ == '__main__':
    # Configure basic logging if you're using app.logger
    # For production, use a more robust logging setup
    import logging
    logging.basicConfig(level=logging.INFO)
    app.logger.info("Starting RadGraph extraction server on port 5000...")
    app.run(host='0.0.0.0', port=5000)