import json
import requests
from typing import Dict, List, Optional, Union

# Add at the top of the file, after imports
VALID_REWARD_COMPONENTS = {
    "radgraphf1",      # RadGraph F1 score
    "cxb14micro",      # CheXbert-14 Micro F1
    "format",          # Format score
    "cxb_bal_acc",     # CheXbert Balanced Accuracy (divided by 14)
    "ratescore",       # RaTE Score
    "all"              # All components
}

def validate_reward_components(components: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
    """Validate and normalize reward component requests.
    
    Args:
        components: List of requested reward components, comma-separated string, or None
        
    Returns:
        Validated list of components or None if input was None
        
    Raises:
        ValueError: If any requested component is invalid
    """
    if components is None:
        return None
        
    # Convert string input to list
    if isinstance(components, str):
        # Handle both "[comp1,comp2]" and "comp1,comp2" formats
        clean_str = components.strip('[]').strip()
        if clean_str == "all":
            components = ["all"]
        else:
            components = [c.strip() for c in clean_str.split(',') if c.strip()]
    
    if not isinstance(components, list):
        raise ValueError(f"Reward components must be a list or string, got {type(components)}")
        
    # Special case for ["all"]
    if components == ["all"]:
        return components
        
    # Validate each component
    invalid_components = [c for c in components if c not in VALID_REWARD_COMPONENTS]
    if invalid_components:
        raise ValueError(
            f"Invalid reward components: {invalid_components}. "
            f"Valid options are: {sorted(VALID_REWARD_COMPONENTS)}"
        )
    
    return components

def rad_compute_score(
    response_str: Union[str, List[str]], 
    ground_truth: Union[str, List[str]], 
    reward_server='http://REWARD_SERVER_HOST:5000/predict', 
    type='cot', 
    requested_reward_components: Optional[List[str]] = None
) -> Union[Dict[str, float], List[Dict[str, float]]]:
    """
    Compute RadGraph scores and ChexBERT scores by sending request to the RadGraph server.
    
    Args:
        response_str (Union[str, List[str]]): The model's response(s)
        ground_truth (Union[str, List[str]]): The reference text(s)
        reward_server (str): URL of the RadGraph server
        type (str): Type of format scoring ('short', 'cot', or 'free_form')
        requested_reward_components (Optional[List[str]]): List of metric names to be summed for the final_score on the server.
        
    Returns:
        Union[Dict[str, float], List[Dict[str, float]]]: Dictionary containing overall, format, and accuracy scores
            If inputs are strings, returns a single dict. If inputs are lists, returns a list of dicts.
    """
    # Handle both single strings and lists
    is_single_input = isinstance(response_str, str)
    
    if is_single_input:
        if not isinstance(ground_truth, str):
            raise ValueError("If response_str is a string, ground_truth must also be a string")
        hyps = [response_str]
        refs = [ground_truth]
    else:
        if not isinstance(ground_truth, list):
            raise ValueError("If response_str is a list, ground_truth must also be a list")
        if len(response_str) != len(ground_truth):
            raise ValueError("response_str and ground_truth lists must have the same length")
        hyps = response_str
        refs = ground_truth
    
    # Validate type
    if type not in {'short', 'cot', 'free_form'}:
        raise ValueError(f"Invalid type '{type}'. Must be one of: 'short', 'cot', 'free_form'")
    
    # Validate components
    validated_components = validate_reward_components(requested_reward_components)
    
    # Prepare data for the server
    senddata = {"hyps": hyps, "refs": refs, "type": type}  # Add 'type' to the request
    if validated_components:
        senddata["requested_reward_components"] = validated_components
    
    # Add debugging prints
    print(f"[DEBUG rad_compute_score] Sending request to: {reward_server}")
    print(f"[DEBUG rad_compute_score] Request data: {senddata}")
    print(f"[DEBUG rad_compute_score] Number of samples: {len(hyps)}")
    if len(hyps) > 0:
        print(f"[DEBUG rad_compute_score] First response text: '{hyps[0][:100]}...' (truncated)")
        print(f"[DEBUG rad_compute_score] First ground truth: '{refs[0][:100]}...' (truncated)")
    
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    try:
        r = requests.post(reward_server, json=senddata, headers=headers)
        # r = requests.post('0.0.0.0:5000/predict', json=senddata, headers=headers)
        
        print(f"[DEBUG rad_compute_score] Server response status: {r.status_code}")
        if r.status_code != 200:
            print(f"[DEBUG rad_compute_score] Error response text: {r.text}")
            print(f"Error from RadGraph server: {r.text}")
            # return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}
            default_result = {"overall": 0.0, "format": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, 
                            "cxb_balanced_accuracy": 0.0, "cxb_accuracy": 0.0, "cxb_14_micro": 0.0, "cxb_14_macro": 0.0, "cxb_5_micro": 0.0, "cxb_5_macro": 0.0,
                            "rate_score": 0.0}
            return default_result if is_single_input else [default_result] * len(hyps)
        
        print(f"[DEBUG rad_compute_score] Raw response: {r.text[:500]}...")  # Truncate for readability
        response = json.loads(r.text)
        if response["result"] != "success":
            print(f"[DEBUG rad_compute_score] Server returned error: {response.get('data', {}).get('message', 'Unknown error')}")
            print(f"Error from RadGraph server: {response['data']['message']}")
            # return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}
            default_result = {"overall": 0.0, "format": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                            "cxb_balanced_accuracy": 0.0, "cxb_accuracy": 0.0, "cxb_14_micro": 0.0, "cxb_14_macro": 0.0, "cxb_5_micro": 0.0, "cxb_5_macro": 0.0,
                            "rate_score": 0.0}
            return default_result if is_single_input else [default_result] * len(hyps)
        
        # Extract data from server response
        predictions = response["data"]["predictions"]
        
        # Handle batch vs single results
        final_scores = predictions["final_scores"]
        format_scores = predictions["format_scores"]
        precision_scores = predictions["precision_radgraph_scores"]
        recall_scores = predictions["recall_radgraph_scores"]
        f1_scores = predictions["f1_radgraph_scores"]
        cxb_balanced_accuracy = predictions["balanced_accuracy_chexbert"]
        cxb_accuracy = predictions["accuracy_chexbert"]
        cxb_14_micro = predictions["micro_chexbert_14"]
        cxb_14_macro = predictions["macro_chexbert_14"]
        cxb_5_micro = predictions["micro_chexbert_5"]
        cxb_5_macro = predictions["macro_chexbert_5"]
        rate_scores = predictions.get("rate_scores", [0.0] * len(hyps))
        
        # Ensure all arrays have the same length
        num_samples = len(hyps)
        
        # Create results for each sample
        results = []
        for i in range(num_samples):
            result = {
                "overall": final_scores[i] if i < len(final_scores) else 0.0,
                "format": format_scores[i] if i < len(format_scores) else 0.0,
                "precision": precision_scores[i] if i < len(precision_scores) else 0.0,
                "recall": recall_scores[i] if i < len(recall_scores) else 0.0,
                "f1": f1_scores[i] if i < len(f1_scores) else 0.0,
                "cxb_balanced_accuracy": cxb_balanced_accuracy[i] if i < len(cxb_balanced_accuracy) else 0.0,
                "cxb_accuracy": cxb_accuracy[i] if i < len(cxb_accuracy) else 0.0,
                "cxb_14_micro": cxb_14_micro[i] if i < len(cxb_14_micro) else 0.0,
                "cxb_14_macro": cxb_14_macro[i] if i < len(cxb_14_macro) else 0.0,
                "cxb_5_micro": cxb_5_micro[i] if i < len(cxb_5_micro) else 0.0,
                "cxb_5_macro": cxb_5_macro[i] if i < len(cxb_5_macro) else 0.0,
                "rate_score": rate_scores[i] if i < len(rate_scores) else 0.0
            }
            results.append(result)
        
        print(f"[DEBUG rad_compute_score] Returning {len(results)} results")
        if is_single_input:
            print(f"[DEBUG rad_compute_score] Single result: {results[0]}")
            return results[0]
        else:
            print(f"[DEBUG rad_compute_score] Batch results: {results}")
            return results
            
    except Exception as e:
        print(f"[DEBUG rad_compute_score] Exception details: {type(e).__name__}: {str(e)}")
        print(f"Exception when calling flask server: {str(e)}")
        # return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}
        default_result = {"overall": 0.0, "format": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                        "cxb_balanced_accuracy": 0.0, "cxb_accuracy": 0.0, "cxb_14_micro": 0.0, "cxb_14_macro": 0.0, "cxb_5_micro": 0.0, "cxb_5_macro": 0.0,
                        "rate_score": 0.0}
        return default_result if is_single_input else [default_result] * len(hyps)

def demo_radgraph_computation():
    """Demo function to test RadGraph score computation with detailed component breakdown and verification"""
    test_cases = [
        {
            "response": "<think> think in progress </think> \\boxed{The patient has pneumonia in the right lung.}",
            "reference": "There is evidence of pneumonia in the right lung.}",
            "type": "cot",
            "components": ["radgraphf1", "ratescore"]  # Example components
        },
        {
            "response": "\\boxed{No significant findings in the chest X-ray.}",
            "reference": "The chest X-ray appears normal.",
            "type": "short",
            "components": ["all"]  # Use all components
        },
        {
            "response": "Mild cardiomegaly observed.",
            "reference": "The heart is slightly enlarged.",
            "type": "free_form",
            "components": ["radgraphf1", "cxb14micro", "cxb_bal_acc", "ratescore"]  # Multiple specific components
        }
    ]
    
    # Component descriptions for equation display
    component_descriptions = {
        "radgraphf1": "RadGraph F1",
        "cxb14micro": "CheXbert-14 Micro",
        "format": "Format Score",
        "cxb_bal_acc": "CheXbert Balanced Acc/14",
        "ratescore": "RaTE Score"
    }
    
    print("Running RadGraph/ChexBERT Demo with Component Breakdown and Verification...\n")
    print("="*80)
    print("TESTING SINGLE INPUT MODE")
    print("="*80)
    
    # Test single input mode
    for i, test_case in enumerate(test_cases, 1):
        response = test_case["response"]
        reference = test_case["reference"]
        type_val = test_case["type"]
        components = test_case["components"]
        
        score = rad_compute_score(response, reference, type=type_val, 
                                requested_reward_components=components)
        
        print(f"Test Case {i}:")
        print(f"  Response:   {response}")
        print(f"  Reference:  {reference}")
        print(f"  Type:       {type_val}")
        print(f"  Requested Components: {components}")
        print("\n  Individual Scores:")
        
        # Print individual component scores
        print(f"    RadGraph F1:                {score['f1']:.4f}")
        print(f"    Format Score:               {score['format']:.4f}")
        print(f"    CheXbert Balanced Acc:      {score['cxb_balanced_accuracy']:.4f}")
        print(f"    CheXbert-14 Micro:          {score['cxb_14_micro']:.4f}")
        print(f"    RaTE Score:                 {score['rate_score']:.4f}")
        
        # Build and display the equation
        print("\n  Final Score Equation:")
        equation_parts = []
        component_values = []  # Store actual values for verification
        
        if components == ["all"]:
            if type_val != 'free_form':
                component_values = [
                    score['f1'],
                    score['cxb_balanced_accuracy']/14.0,
                    score['rate_score'],
                    score['format']
                ]
                equation_parts = [
                    f"RadGraph F1 ({score['f1']:.4f})",
                    f"CheXbert Balanced Acc/14 ({score['cxb_balanced_accuracy']/14:.4f})",
                    f"RaTE Score ({score['rate_score']:.4f})",
                    f"Format Score ({score['format']:.4f})"
                ]
            else:
                component_values = [
                    score['f1'],
                    score['cxb_balanced_accuracy']/14.0,
                    score['rate_score']
                ]
                equation_parts = [
                    f"RadGraph F1 ({score['f1']:.4f})",
                    f"CheXbert Balanced Acc/14 ({score['cxb_balanced_accuracy']/14:.4f})",
                    f"RaTE Score ({score['rate_score']:.4f})"
                ]
        else:
            for comp in components:
                if comp == "radgraphf1":
                    component_values.append(score['f1'])
                    equation_parts.append(f"RadGraph F1 ({score['f1']:.4f})")
                elif comp == "cxb14micro":
                    component_values.append(score['cxb_14_micro'])
                    equation_parts.append(f"CheXbert-14 Micro ({score['cxb_14_micro']:.4f})")
                elif comp == "format" and type_val != 'free_form':
                    component_values.append(score['format'])
                    equation_parts.append(f"Format Score ({score['format']:.4f})")
                elif comp == "cxb_bal_acc":
                    component_values.append(score['cxb_balanced_accuracy']/14.0)
                    equation_parts.append(f"CheXbert Balanced Acc/14 ({score['cxb_balanced_accuracy']/14:.4f})")
                elif comp == "ratescore":
                    component_values.append(score['rate_score'])
                    equation_parts.append(f"RaTE Score ({score['rate_score']:.4f})")
        
        equation = " + ".join(equation_parts)
        calculated_sum = sum(component_values)
        
        print(f"    {equation} = {calculated_sum:.4f}")
        print(f"    Server returned overall score: {score['overall']:.4f}")
        
        # Verify if the calculated sum matches the overall score
        if abs(calculated_sum - score['overall']) < 1e-6:  # Using small epsilon for float comparison
            print("    ✓ Verification: Calculated sum matches server's overall score")
        else:
            print(f"    ⚠ Verification: Mismatch between calculated sum ({calculated_sum:.4f}) "
                  f"and server's overall score ({score['overall']:.4f})")
        
        print("\n" + "="*80 + "\n")

    print("="*80)
    print("TESTING BATCH INPUT MODE")
    print("="*80)
    
    # Test batch input mode
    responses = [case["response"] for case in test_cases]
    references = [case["reference"] for case in test_cases]
    
    print(f"Testing batch processing with {len(responses)} samples")
    print("Using 'free_form' type and ['radgraphf1'] components for batch test")
    
    try:
        batch_scores = rad_compute_score(
            responses, references, 
            type='free_form', 
            requested_reward_components=['radgraphf1']
        )
        
        print(f"Batch processing successful! Got {len(batch_scores)} results:")
        for i, score in enumerate(batch_scores):
            print(f"  Sample {i+1}: overall={score['overall']:.4f}, f1={score['f1']:.4f}, format={score['format']:.4f}")
            
        print("\n✓ Batch processing test passed!")
        
    except Exception as e:
        print(f"✗ Batch processing test failed: {str(e)}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    demo_radgraph_computation()