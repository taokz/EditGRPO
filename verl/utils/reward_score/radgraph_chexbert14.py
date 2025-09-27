import json
import requests
from typing import Dict

def radgraph_chexbert14_compute_score(response_str: str, ground_truth: str, reward_server='http://REWARD_SERVER_HOST:5000/predict', type='cot') -> Dict[str, float]:
    """
    Compute RadGraph scores and ChexBERT scores by sending request to the RadGraph server.
    
    Args:
        response_str (str): The model's response
        ground_truth (str): The reference text
        reward_server (str): URL of the RadGraph server
        type (str): Type of format scoring ('short' or 'cot')
        
    Returns:
        Dict[str, float]: Dictionary containing overall, format, and accuracy scores
    """
    # Prepare data for the server
    hyps = [response_str]
    refs = [ground_truth]
    
    senddata = {"hyps": hyps, "refs": refs, "type": type}  # Add 'type' to the request
    
    headers = {'Content-Type': 'application/json; charset=utf-8'}
    try:
        r = requests.post(reward_server, json=senddata, headers=headers)
        # r = requests.post('0.0.0.0:5000/predict', json=senddata, headers=headers)
        
        if r.status_code != 200:
            print(f"Error from RadGraph server: {r.text}")
            # return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}
            return {"overall": 0.0, "format": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, 
                    "cxb_accuracy": 0.0, "cxb_14_micro": 0.0, "cxb_14_macro": 0.0, "cxb_5_micro": 0.0, "cxb_5_macro": 0.0}
        
        response = json.loads(r.text)
        if response["result"] != "success":
            print(f"Error from RadGraph server: {response['data']['message']}")
            # return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}
            return {"overall": 0.0, "format": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "cxb_accuracy": 0.0, "cxb_14_micro": 0.0, "cxb_14_macro": 0.0, "cxb_5_micro": 0.0, "cxb_5_macro": 0.0}
        
        # Return the scores
        # return {
        #     "overall": response["data"]["predictions"]["final_scores"][0],
        #     "format": response["data"]["predictions"]["format_scores"][0],
        #     "accuracy": response["data"]["predictions"]["radgraph_scores"][0],
        # }
        return {
            "overall": response["data"]["predictions"]["final_scores"][0],
            "format": response["data"]["predictions"]["format_scores"][0],
            "precision": response["data"]["predictions"]["precision_radgraph_scores"][0],
            "recall": response["data"]["predictions"]["recall_radgraph_scores"][0],
            "f1": response["data"]["predictions"]["f1_radgraph_scores"][0],
            "cxb_accuracy": response["data"]["predictions"]["accuracy_chexbert"][0],
            "cxb_14_micro": response["data"]["predictions"]["micro_chexbert_14"][0],
            "cxb_14_macro": response["data"]["predictions"]["macro_chexbert_14"][0],
            "cxb_5_micro": response["data"]["predictions"]["micro_chexbert_5"][0],
            "cxb_5_macro": response["data"]["predictions"]["macro_chexbert_5"][0]
        }   
    except Exception as e:
        print(f"Exception when calling RadGraph/ChexBERT server: {str(e)}")
        # return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}
        return {"overall": 0.0, "format": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "cxb_14_micro": 0.0, "cxb_14_macro": 0.0, "cxb_5_micro": 0.0, "cxb_5_macro": 0.0}

def demo_radgraph_computation():
    """Demo function to test RadGraph score computation"""
    test_cases = [
        {
            "response": "<think> think in progress </think> \\boxed{The patient has pneumonia in the right lung.}",
            "reference": "There is evidence of pneumonia in the right lung.}"
        },
        {
            "response": "\\boxed{No significant findings in the chest X-ray.}",
            "reference": "The chest X-ray appears normal."
        },
        {
            "response": "Mild cardiomegaly observed.",
            "reference": "The heart is slightly enlarged."
        }
    ]
    
    print("Running RadGraph/ChexBERT Demo...\n")
    for i, test_case in enumerate(test_cases, 1):
        response = test_case["response"]
        reference = test_case["reference"]
        score = radgraph_chexbert14_compute_score(response, reference)
        print(f"Test Case {i}:")
        print(f"  Response:   {response}")
        print(f"  Reference:  {reference}")
        print("  Scores:")
        for key, value in score.items():
            print(f"    {key}: {value}")
        print()  # Add an empty line for separation

if __name__ == "__main__":
    demo_radgraph_computation()