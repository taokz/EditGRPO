# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections import defaultdict
from typing import Any, Callable, Dict, Tuple, TypedDict, Optional, List, Union

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from ...utils.reward_score import math_compute_score, r1v_compute_score, bleu4_compute_score
from ...utils.reward_score.rad_scores import rad_compute_score, VALID_REWARD_COMPONENTS, validate_reward_components
from ...utils.reward_score.green_scores import green_compute_score, VALID_GREEN_REWARD_COMPONENTS, validate_green_reward_components


class RewardScore(TypedDict):
    overall: float
    format: float
    accuracy: float

class Rad_RewardScore(TypedDict):
    overall: float
    format: float
    precision: float
    recall: float
    f1: float
    cxb_balanced_accuracy: float
    cxb_accuracy: float
    cxb_14_micro: float
    cxb_14_macro: float
    cxb_5_micro: float
    cxb_5_macro: float
    rate_score: float

class Green_RewardScore(TypedDict):
    overall: float
    format: float
    green_mean: float
    green_std: float


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer,
                 compute_score: str,
                 reward_server_ip: str = 'http://REWARD_SERVER_HOST:5000/predict',
                 prompt_type: str = 'cot',
                 return_ground_truth: bool = False,
                 server_aggregation_components: Optional[List[str]] = None,
                 reward_calc_type: str = 'sequential'):
        
        self.reward_calc_type = reward_calc_type
        self.compute_score_type = compute_score
        
        # Add debug prints
        print(f"[DEBUG] CustomRewardManager initialized with:")
        print(f"  - compute_score: {compute_score}")
        print(f"  - reward_server_ip: {reward_server_ip}")
        print(f"  - prompt_type: {prompt_type}")
        print(f"  - server_aggregation_components: {server_aggregation_components}")
        print(f"  - reward_calc_type: {reward_calc_type}")
        
        # Validate prompt type
        if prompt_type not in {'short', 'cot', 'free_form'}:
            raise ValueError(f"Invalid prompt_type '{prompt_type}'. Must be one of: 'short', 'cot', 'free_form'")
        
        # Validate components based on compute_score type
        if compute_score == "rad_server":
            validated_components = validate_reward_components(server_aggregation_components)
        elif compute_score == "green_server":
            validated_components = validate_green_reward_components(server_aggregation_components)
        else:
            validated_components = None
        
        print(f"[DEBUG] Validated components: {validated_components}")
        
        self.tokenizer = tokenizer
        self.return_ground_truth = return_ground_truth
        
        self._scoring_function: Callable[[str, str], Dict[str, float]] = None
        
        if compute_score == "math":
            self._scoring_function = math_compute_score
        elif compute_score == "r1v":
            self._scoring_function = r1v_compute_score
        elif compute_score == "bleu4":
            def _bleu4_wrapper(response_str: str, ground_truth: str) -> RewardScore:
                return bleu4_compute_score(response_str, ground_truth)
            self._scoring_function = _bleu4_wrapper
        elif compute_score == "rad_server": 
            def _server_score_wrapper(response_str: str, ground_truth: str) -> Rad_RewardScore:
                return rad_compute_score(
                    response_str, ground_truth,
                    reward_server=reward_server_ip, 
                    type=prompt_type,
                    requested_reward_components=validated_components
                )
            self._scoring_function = _server_score_wrapper
        elif compute_score == "green_server":
            def _green_server_wrapper(response_str: str, ground_truth: str) -> Green_RewardScore:
                print(f"[DEBUG] Calling green_compute_score with:")
                print(f"  - response_str: '{response_str[:100] if isinstance(response_str, str) else str(response_str)[:100]}...' (truncated)")
                print(f"  - ground_truth: '{ground_truth[:100] if isinstance(ground_truth, str) else str(ground_truth)[:100]}...' (truncated)")
                print(f"  - reward_server: {reward_server_ip}")
                print(f"  - type: {prompt_type}")
                print(f"  - requested_reward_components: {validated_components}")
                
                result = green_compute_score(
                    response_str, ground_truth,
                    reward_server=reward_server_ip,
                    type=prompt_type,
                    requested_reward_components=validated_components
                )
                print(f"[DEBUG] green_compute_score returned: {result}")
                return result
            self._scoring_function = _green_server_wrapper
        else:
            raise NotImplementedError(f"compute_score string '{compute_score}' is not implemented. Use 'math', 'r1v', 'bleu4', 'rad_server', or 'green_server' (with server_aggregation_components)." )

    def __call__(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, Any], Optional[List[str]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        ground_truths: Optional[List[str]] = [] if self.return_ground_truth else None

        if self._scoring_function is None:
            raise RuntimeError("Scoring function was not initialized properly.")

        print(f"[DEBUG] Processing {len(data)} samples with reward_calc_type: {self.reward_calc_type}")

        if self.reward_calc_type == 'sequential':
            # Original sequential processing
            for i in range(len(data)):
                data_item = data[i]
                response_ids = data_item.batch["responses"]
                response_mask = data_item.batch["response_mask"]
                valid_response_length = response_mask.sum()
                valid_response_ids = response_ids[:valid_response_length]

                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                ground_truth_str = data_item.non_tensor_batch["ground_truth"]

                print(f"[DEBUG] Processing sample {i+1}/{len(data)}")
                score_dict = self._scoring_function(response_str, ground_truth_str)
                
                reward_tensor[i, valid_response_length - 1] = score_dict["overall"]
                for key, value in score_dict.items():
                    reward_metrics[key].append(value)

                if self.return_ground_truth and ground_truths is not None:
                    ground_truths.append(ground_truth_str)
                    
        elif self.reward_calc_type == 'batch':
            # Batch processing - collect all responses and ground truths first
            response_strings = []
            ground_truth_strings = []
            response_lengths = []
            
            for i in range(len(data)):
                data_item = data[i]
                response_ids = data_item.batch["responses"]
                response_mask = data_item.batch["response_mask"]
                valid_response_length = response_mask.sum()
                valid_response_ids = response_ids[:valid_response_length]

                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                ground_truth_str = data_item.non_tensor_batch["ground_truth"]
                
                response_strings.append(response_str)
                ground_truth_strings.append(ground_truth_str)
                response_lengths.append(valid_response_length)
                
                if self.return_ground_truth and ground_truths is not None:
                    ground_truths.append(ground_truth_str)
            
            # Process all at once - both rad_server and green_server now support batch processing
            try:
                # Try batch processing first
                print(f"[DEBUG] Attempting batch processing for {self.compute_score_type}")
                score_dicts = self._scoring_function(response_strings, ground_truth_strings)
                
                # Ensure score_dicts is a list
                if not isinstance(score_dicts, list):
                    # This handles the case where a single dict is returned for batch input
                    # (shouldn't happen with properly implemented batch functions, but just in case)
                    score_dicts = [score_dicts] * len(response_strings)
                    
                print(f"[DEBUG] Batch processing successful, got {len(score_dicts)} results")
                
            except (TypeError, ValueError) as e:
                print(f"[DEBUG] Batch processing failed ({e}), falling back to sequential")
                # Fallback to sequential processing if batch is not supported
                score_dicts = []
                for response_str, ground_truth_str in zip(response_strings, ground_truth_strings):
                    score_dicts.append(self._scoring_function(response_str, ground_truth_str))
            
            # Assign rewards
            for i, (score_dict, valid_response_length) in enumerate(zip(score_dicts, response_lengths)):
                reward_tensor[i, valid_response_length - 1] = score_dict["overall"]
                for key, value in score_dict.items():
                    reward_metrics[key].append(value)
        else:
            raise ValueError(f"Invalid reward_calc_type '{self.reward_calc_type}'. Must be 'sequential' or 'batch'.")
        
        print(f"[DEBUG] Final reward_metrics summary: {dict(reward_metrics)}")
        return reward_tensor, reward_metrics, ground_truths
