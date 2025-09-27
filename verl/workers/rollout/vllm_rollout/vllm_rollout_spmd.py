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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
"""

import os
from contextlib import contextmanager
from typing import Any, List, Union

import numpy as np
import torch
import torch.distributed
import requests
import json
from tensordict import TensorDict
from transformers import PreTrainedTokenizer
from vllm import LLM, RequestOutput, SamplingParams

from ....protocol import DataProto
from ....utils import torch_functional as VF
from ....utils.torch_dtypes import PrecisionType
from ....utils.radgraph_extractor import process_annotations
from ....utils.RaTEscore_extractor import process_text_with_server_analysis
from ..base import BaseRollout
from ..config import RolloutConfig
# from radgraph import RadGraph

import random


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: RolloutConfig, tokenizer: PreTrainedTokenizer):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if not config.enforce_eager and config.free_cache_engine:
            raise ValueError("CUDA graph should be disabled when `free_cache_engine` is True.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        vllm_init_kwargs = {}
        if config.limit_images > 0:
            vllm_init_kwargs = {"limit_mm_per_prompt": {"image": config.limit_images}}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=config.enforce_eager,
            max_model_len=config.prompt_length + config.response_length,
            max_num_batched_tokens=config.max_num_batched_tokens,
            enable_sleep_mode=True,
            distributed_executor_backend="external_launcher",
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            disable_log_stats=config.disable_log_stats,
            enable_chunked_prefill=config.enable_chunked_prefill,
            seed=config.seed,
            **vllm_init_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {"max_tokens": config.response_length, "detokenize": False}
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

        # Store the seed for consistent random operations
        self.seed = config.seed
        # Create a dedicated random number generator for paragraph sampling
        self.paragraph_rng = random.Random(self.seed)

    def _clean_text_and_tokenize(self, text: str, max_length: int = None) -> List[int]:
        """
        Clean text of literal <|endoftext|> strings and tokenize safely.
        
        Args:
            text: Text to clean and tokenize
            max_length: Maximum length to truncate tokens to
            
        Returns:
            List of token IDs with EOS tokens removed
        """
        # Ensure text is a string
        text = str(text).strip()
        
        # Clean the text of all EOS variations
        text = self._clean_eos_from_text(text)
        
        # Tokenize without special tokens
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Count EOS tokens before removal for debugging
        original_eos_count = 0
        if self.tokenizer.eos_token_id is not None:
            original_eos_count = tokens.count(self.tokenizer.eos_token_id)
            tokens = [token for token in tokens if token != self.tokenizer.eos_token_id]
            
            if self.rank == 0 and original_eos_count > 0:
                print(f"DEBUG: Removed {original_eos_count} EOS token IDs from tokenized text")
        
        # Truncate if max_length is specified
        if max_length is not None:
            tokens = tokens[:max_length]
            
        return tokens

    def _clean_eos_from_text(self, text: str) -> str:
        """
        Remove all variations of EOS tokens from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        original_text = text
        
        # Remove literal <|endoftext|> strings from the text
        if self.tokenizer.eos_token:
            text = text.replace(self.tokenizer.eos_token, "")
        
        # Common EOS token variations across different models
        eos_variations = [
            "<|endoftext|>", "<|end_of_text|>", "</s>", "<eos>", 
            "<|eot_id|>", "[EOS]", "[END]", "<|im_end|>",
            # Also handle with spaces/newlines around them
            " <|endoftext|>", "<|endoftext|> ", " <|endoftext|> ",
            "\n<|endoftext|>", "<|endoftext|>\n", "\n<|endoftext|>\n",
            " </s>", "</s> ", " </s> ",
            "\n</s>", "</s>\n", "\n</s>\n"
        ]
        
        # Use regex for more thorough cleaning
        import re
        # Pattern to match EOS tokens with optional spaces/newlines around them
        eos_pattern = r'[\s]*(<\|endoftext\|>|<\|end_of_text\|>|</s>|<eos>|<\|eot_id\|>|\[EOS\]|\[END\]|<\|im_end\|>)[\s]*'
        text = re.sub(eos_pattern, ' ', text)
        
        # Also do string replacement as fallback
        for eos_var in eos_variations:
            text = text.replace(eos_var, " ")
        
        # Also handle multiple consecutive EOS tokens
        # Sometimes models generate things like <|endoftext|><|endoftext|>
        # Keep replacing until no more changes occur
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        while iteration < max_iterations:
            old_len = len(text)
            text = re.sub(eos_pattern, ' ', text)
            for eos_var in eos_variations[:8]:  # Check main variations
                text = text.replace(eos_var, " ")
            if len(text) == old_len:  # No more changes
                break
            iteration += 1
        
        # Clean up any resulting double spaces or newlines
        text = ' '.join(text.split())
        
        # Additional cleanup for edge cases
        # Remove any remaining partial EOS tokens that might be broken
        partial_eos_patterns = ["<|endof", "text|>", "<|end", "oftext|>"]
        for pattern in partial_eos_patterns:
            if pattern in text:
                # This might indicate a broken EOS token
                if self.rank == 0:
                    print(f"WARNING: Found partial EOS pattern '{pattern}' in text, this might indicate tokenization issues")
        
        if self.rank == 0 and original_text != text:
            # Count how many EOS tokens were removed
            eos_count = original_text.count("<|endoftext|>") + original_text.count("</s>")
            if eos_count > 1:
                print(f"DEBUG: Cleaned {eos_count} EOS tokens from text (including {eos_count - 1} duplicates)")
            else:
                print(f"DEBUG: Cleaned EOS tokens from text. Original length: {len(original_text)}, Cleaned length: {len(text)}")
            
        return text.strip()

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
            ):
                vllm_inputs.append({"prompt_token_ids": list(raw_prompt_ids), "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        """
        algorithm.force_gt_number functionality
        """
        # Check if we should use ground truth for some samples
        gt_indices = prompts.meta_info.get("gt_indices", [])
        force_gt = len(gt_indices) > 0
        original_batch_size = prompts.meta_info.get("original_batch_size", None)

        """
        rollout.extractor functionality
        """
        # Whether to activate token & label extractio54n from rollout responses and ground truth
        extractor_flag = prompts.meta_info.get("extract_flag", False)
        extractor_type = prompts.meta_info.get("extractor_type", None)
        extractor_server = prompts.meta_info.get("extractor_server", None)
        extractor_num_ops_per_para = prompts.meta_info.get("extractor_num_ops_per_para", None)
        extractor_num_para = prompts.meta_info.get("extractor_num_para", None)
        extractor_similarity_threshold = prompts.meta_info.get("extractor_similarity_threshold", 0.6)

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            completions: List[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=(self.rank == 0)
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            ### For debugging
            if self.rank == 0:
                print(f"Number of completions: {len(completions)}")
                print(f"Number of response_ids: {len(response_ids)}")  # Should be batch_size * n
            
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)
            
            # DEBUG: Check if vLLM generated EOS tokens
            if self.rank == 0 and self.tokenizer.eos_token_id is not None:
                original_eos_count = (response_ids == self.tokenizer.eos_token_id).sum().item()
                if original_eos_count > 0:
                    print(f"DEBUG: vLLM generated {original_eos_count} EOS tokens in response_ids")

            # Overwrite with ground truth responses if needed
            if force_gt and "ground_truth" in non_tensor_batch:
                # Convert global gt_indices to local indices relative to this worker's batch portion
                current_batch_size = input_ids.size(0)
                
                # Find which ground truth indices fall within this worker's batch
                valid_local_indices = []
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                
                if world_size > 1:
                    # For distributed training, find which indices belong to this rank
                    if self.rank == 0:
                        print(f"Worker process rank {rank}: Current batch size {current_batch_size}, Original batch size {original_batch_size if original_batch_size else 'unknown'}")
                    
                    # In distributed training, samples are divided among workers
                    # For FSDP and tensor parallelism, each worker gets a subset of the samples
                    for i, global_idx in enumerate(gt_indices):
                        # For FSDP and tensor parallelism, the batch is sharded by sample indices
                        # We need to check if the global index is in this worker's local shard
                        local_idx = global_idx % current_batch_size  # Local index within this worker's batch
                        worker_id = global_idx // current_batch_size  # Which worker would get this sample
                        
                        # If this worker is responsible for this sample
                        if worker_id == rank % (world_size // self.config.tensor_parallel_size):
                            valid_local_indices.append((i, local_idx))
                else:
                    # For single GPU, all indices are local
                    for i, gt_idx in enumerate(gt_indices):
                        if gt_idx < current_batch_size:
                            valid_local_indices.append((i, gt_idx))
                
                # FIXED: Check only the valid local indices count against available ground truths
                if len(valid_local_indices) > len(non_tensor_batch["ground_truth"]):
                    raise ValueError(f"Number of local gt_indices ({len(valid_local_indices)}) cannot be greater than the number of provided ground truths ({len(non_tensor_batch['ground_truth'])}) on worker {rank}.")
                
                if self.rank == 0:
                    print(f"Worker process rank {rank}: Processing {len(valid_local_indices)} ground truth samples from {len(gt_indices)} total gt_indices")
                    
                # Create a random generator for candidate selection with a different seed
                candidate_generator = torch.Generator()
                candidate_generator.manual_seed(self.seed + rank)  # Different seed per worker
                
                # Track which indices were actually replaced for verification
                actual_replace_indices = []
                
                for i, gt_idx in valid_local_indices:
                    if i >= len(non_tensor_batch["ground_truth"]):
                         # This case should be prevented by the check above, but added for safety.
                         print(f"Warning: Not enough ground truth strings provided. Skipping gt_index {gt_idx}.")
                         continue

                    # Tokenize the ground truth exactly as the model would during generation
                    gt_text = str(non_tensor_batch["ground_truth"][i]) # Use index 'i' matching gt_indices order
                    # Get token IDs without any special tokens to match generation output
                    gt_tokens = self._clean_text_and_tokenize(gt_text, self.config.response_length)
                    length = len(gt_tokens)  # length is already capped by _clean_text_and_tokenize

                    # Create padded ground truth tensor
                    gt_response = torch.full(
                        (self.config.response_length,),
                        self.pad_token_id,
                        dtype=input_ids.dtype,
                        device=input_ids.device
                    )
                    gt_response[:length] = torch.tensor(
                        gt_tokens[:length],
                        dtype=input_ids.dtype,
                        device=input_ids.device
                    )

                    # FIXED: Randomly select which candidate to replace instead of always the first
                    # Choose a random candidate (0 to n-1) for this prompt
                    random_candidate_idx = torch.randint(0, self.sampling_params.n, (1,), generator=candidate_generator).item()
                    replace_idx = gt_idx * self.sampling_params.n + random_candidate_idx
                    
                    if replace_idx >= response_ids.shape[0]:
                         print(f"Warning: Calculated replace index {replace_idx} is out of bounds for response_ids shape {response_ids.shape}. Skipping.")
                         continue

                    if self.rank == 0:
                        print(f"Replacing candidate {random_candidate_idx} (out of {self.sampling_params.n}) for prompt {gt_idx} at response index {replace_idx}")

                    response_ids[replace_idx] = gt_response
                    actual_replace_indices.append((i, gt_idx, replace_idx, gt_text))

            # ALL RANKS should store the completions for comparison
            # self.rank == 0 should do the verification (add in the following previously)
            if force_gt and "ground_truth" in non_tensor_batch:
                original_completions = [output.token_ids for completion in completions for output in completion.outputs]
                original_completions = VF.pad_2d_list_to_length(
                    original_completions, self.pad_token_id, max_length=self.config.response_length
                ).to(input_ids.device)
                
                # Verification logic only on rank 0
                if self.rank == 0:
                    verification_results = []
                    for i, gt_idx, replace_idx, gt_text in actual_replace_indices:
                        if replace_idx >= response_ids.shape[0]:
                            continue
                            
                        # Check if the response was modified
                        original_response = original_completions[replace_idx]
                        modified_response = response_ids[replace_idx]
                        
                        # Compare the responses
                        is_modified = not torch.equal(original_response, modified_response)
                        
                        verification_results.append({
                            "gt_idx": gt_idx,
                            "replace_idx": replace_idx,
                            "is_modified": is_modified,
                            "gt_text": gt_text
                        })
                    
                    # Log verification results
                    print("\n=== Ground Truth Application Verification ===")
                    print(f"Total ground truth samples to apply: {len(actual_replace_indices)}")
                    print(f"Successfully verified: {len(verification_results)}")
                    
                    # Count how many were actually modified
                    modified_count = sum(1 for result in verification_results if result["is_modified"])
                    print(f"Successfully modified: {modified_count}/{len(verification_results)}")
                    
                    # Log details for any that weren't modified
                    if modified_count < len(verification_results):
                        print("\nSamples that were not modified:")
                        for result in verification_results:
                            if not result["is_modified"]:
                                print(f"  - GT index {result['gt_idx']} (replace_idx: {result['replace_idx']}): {result['gt_text'][:50]}...")
                    
                    print("===========================================\n")

            if extractor_flag:
                # (1) Get the text of the generated responses
                gen_texts = []
                for ids in response_ids:
                    valid_ids = ids[ids != self.pad_token_id].tolist()
                    gen_text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                    # Clean any remaining EOS tokens that were decoded as literal text
                    gen_text = self._clean_eos_from_text(gen_text)
                    gen_texts.append(gen_text)
                
                # Diagnostic: Check if any generated texts originally had EOS tokens
                if self.rank == 0:
                    eos_in_generated = 0
                    for i, ids in enumerate(response_ids):
                        valid_ids = ids[ids != self.pad_token_id].tolist()
                        raw_text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                        if "<|endoftext|>" in raw_text or "</s>" in raw_text:
                            eos_in_generated += 1
                            if i < 3:  # Show first few examples
                                print(f"DEBUG: Generated text {i} contains EOS tokens BEFORE modification: {raw_text[:200]}...")
                    
                    if eos_in_generated > 0:
                        print(f"WARNING: {eos_in_generated}/{len(response_ids)} generated texts contain literal EOS tokens")
                        print("This suggests the model was trained with literal EOS tokens in the data")
                
                # Strict ground truth checking
                if "ground_truth" not in non_tensor_batch:
                    raise ValueError("Ground truth is required for extraction but not provided in non_tensor_batch")
                
                gt_texts = non_tensor_batch["ground_truth"]
                
                # Check for empty or None ground truths
                if gt_texts is None:
                    raise ValueError("Ground truth is None")
                if len(gt_texts) == 0:
                    raise ValueError("Ground truth list is empty")
                if any(not gt or str(gt).strip() == "" for gt in gt_texts): # Ensure conversion to str for strip
                    raise ValueError("One or more ground truth strings are empty or only whitespace")
                
                if self.rank == 0:
                    print("\nGround truth debug:")
                    print(f"Type of gt_texts: {type(gt_texts)}")
                    print(f"Length of gt_texts (number of original prompts): {len(gt_texts)}")
                    print(f"Number of generations per prompt (n): {self.sampling_params.n}")
                    print(f"Total generations (batch_size * n): {len(gen_texts)}")
                
                # Expand ground truths to match the number of generated texts
                # Each of the 'n' generations for a prompt uses the same ground truth from its original prompt
                processed_gt_texts = [str(gt) for gt in gt_texts] # Ensure all are strings
                
                # Clean EOS tokens from ground truth texts to prevent them from being injected
                processed_gt_texts = [self._clean_eos_from_text(gt) for gt in processed_gt_texts]
                
                # Additional validation to ensure no EOS tokens remain
                for i, gt in enumerate(processed_gt_texts):
                    if any(eos in gt for eos in ["<|endoftext|>", "</s>", "<eos>", "<|eot_id|>"]):
                        if self.rank == 0:
                            print(f"WARNING: Ground truth {i} still contains EOS tokens after cleaning: {gt[:100]}...")
                        # Force another cleaning pass
                        processed_gt_texts[i] = self._clean_eos_from_text(gt)
                
                # Debug logging for EOS cleaning in ground truth
                if self.rank == 0:
                    gt_eos_cleaned = sum(1 for i, (orig, clean) in enumerate(zip([str(gt) for gt in gt_texts], processed_gt_texts)) if orig != clean)
                    if gt_eos_cleaned > 0:
                        print(f"DEBUG: Cleaned EOS tokens from {gt_eos_cleaned} ground truth texts before sending to server")
                
                num_original_prompts = len(processed_gt_texts)
                num_gens_per_prompt = self.sampling_params.n
                
                expanded_gt_texts_for_server = []
                if num_gens_per_prompt > 0:
                    for i in range(num_original_prompts):
                        expanded_gt_texts_for_server.extend([processed_gt_texts[i]] * num_gens_per_prompt)
                
                # Safety check for length consistency
                if len(gen_texts) != len(expanded_gt_texts_for_server):
                    if self.rank == 0:
                        print(f"Warning: Mismatch between number of generated texts ({len(gen_texts)}) "
                              f"and number of expanded ground truths ({len(expanded_gt_texts_for_server)}). "
                              "This may lead to incorrect behavior in the extractor.")
                        print(f"  Details: num_original_prompts={num_original_prompts}, num_gens_per_prompt={num_gens_per_prompt}")
                    # Attempt to reconcile if possible, or raise error, or truncate.
                    # For now, this warning highlights a potential issue if lengths don't match as expected.
                    # One common cause could be if gen_texts isn't exactly num_original_prompts * num_gens_per_prompt.

                data = {
                    "generations": gen_texts,
                    "ground_truth": expanded_gt_texts_for_server # Now a list parallel to generations
                }
                
                if self.rank == 0: # Logging for Issue 4
                    print("\n=== Extractor Request Data (Sample) ===")
                    print(f"Number of generations sent: {len(data['generations'])}")
                    print(f"Number of ground truths sent: {len(data['ground_truth'])}")
                    
                    # Determine the number of original prompts based on the unexpanded gt_texts
                    # gt_texts was defined earlier as: gt_texts = non_tensor_batch["ground_truth"]
                    num_original_prompts_in_batch = len(gt_texts) 
                    gens_per_prompt_n = self.sampling_params.n
                    
                    prompts_to_display_in_log = min(2, num_original_prompts_in_batch) # Log for up to 2 original prompts

                    for p_idx in range(prompts_to_display_in_log):
                        print(f"  --- Samples for Original Prompt Index {p_idx} (out of {num_original_prompts_in_batch}) ---")
                        # Log for up to 'n' candidates, but cap at 3 for brevity, ensuring at least 2 if n>=2
                        max_candidates_to_log = min(gens_per_prompt_n, 3) 
                        if gens_per_prompt_n == 1: # if n=1, show that 1 candidate
                            candidates_to_display_for_this_prompt = 1
                        else: # if n > 1, show at least 2, up to max_candidates_to_log
                            candidates_to_display_for_this_prompt = min(max(2, gens_per_prompt_n if gens_per_prompt_n < max_candidates_to_log else max_candidates_to_log) , gens_per_prompt_n)
                        
                        for c_idx in range(candidates_to_display_for_this_prompt):
                            actual_flat_idx = p_idx * gens_per_prompt_n + c_idx
                            
                            if actual_flat_idx < len(data['generations']):
                                gen_sample = data['generations'][actual_flat_idx]
                                gen_display = gen_sample[:100] + "..." if len(gen_sample) > 100 else gen_sample
                                
                                gt_display = "(GT missing or index out of bounds)"
                                if actual_flat_idx < len(data['ground_truth']):
                                    gt_text_sample = data['ground_truth'][actual_flat_idx]
                                    gt_display = gt_text_sample[:100] + "..." if len(gt_text_sample) > 100 else gt_text_sample
                                
                                print(f"    Gen (Cand {c_idx + 1}/{gens_per_prompt_n}): {gen_display}")
                                print(f"    GT  (Cand {c_idx + 1}/{gens_per_prompt_n}): {gt_display}") # GT is the same for all candidates of one original prompt
                            else:
                                print(f"    (Log: Generation index {actual_flat_idx} out of bounds for prompt {p_idx}, cand {c_idx})")
                    print("=======================================\n")

                if extractor_type == "ratescore":
                    # Store original texts for verification
                    original_texts = gen_texts.copy()
                    # Store original response_ids for verification
                    original_response_ids = response_ids.clone()
                    
                    # Call RaTEscore extractor server directly (similar to RadGraph approach)
                    headers = {'Content-Type': 'application/json'}
                    
                    # Log request details
                    if self.rank == 0:
                        print("\nDEBUG: About to call RaTEscore server")
                        print(f"DEBUG: Server URL: {extractor_server}")
                        print(f"DEBUG: Request headers: {headers}")
                        if len(data['generations']) > 0:
                            print(f"DEBUG: Sample generation (first 100 chars): {data['generations'][0][:100]}")
                        if len(data['ground_truth']) > 0:
                            print(f"DEBUG: Sample ground truth (first 100 chars): {data['ground_truth'][0][:100]}")
                        
                    # Send the request to the server
                    response = requests.post(extractor_server or "http://localhost:5001/analyze", json=data, headers=headers)
                    
                    # Log response details
                    if self.rank == 0:
                        print(f"\nDEBUG: RaTEscore server response status: {response.status_code}")
                        print(f"DEBUG: Response headers: {dict(response.headers)}")
                        if response.status_code != 200:
                            print(f"DEBUG: Error response text: {response.text}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Log the full structure of the response
                        if self.rank == 0:
                            print(f"\nDEBUG: Full response structure:")
                            print(f"DEBUG: Result keys: {list(result.keys())}")
                            for key in result:
                                print(f"DEBUG: {key} type: {type(result[key])}")
                                if isinstance(result[key], dict):
                                    print(f"DEBUG: {key} keys: {list(result[key].keys())}")
                        
                        # Get the analysis results from the response
                        data_results = result.get("data", {})
                        predictions = data_results.get("predictions", {})
                        analysis_results = predictions.get("analysis_results", [])
                        
                        if self.rank == 0:
                            print(f"\nDEBUG: Analysis results structure:")
                            print(f"DEBUG: Analysis results type: {type(analysis_results)}")
                            print(f"DEBUG: Analysis results length: {len(analysis_results)}")
                            if analysis_results and len(analysis_results) > 0:
                                print(f"DEBUG: First analysis result type: {type(analysis_results[0])}")
                                if isinstance(analysis_results[0], dict):
                                    print(f"DEBUG: First analysis result keys: {list(analysis_results[0].keys())}")
                                    for key in analysis_results[0]:
                                        value = analysis_results[0][key]
                                        print(f"DEBUG: First result '{key}' type: {type(value)}")
                                        if isinstance(value, list):
                                            print(f"DEBUG: First result '{key}' length: {len(value)}")
                                            if len(value) > 0:
                                                print(f"DEBUG: First result '{key}' first item type: {type(value[0])}")
                                                if isinstance(value[0], dict):
                                                    print(f"DEBUG: First result '{key}' first item keys: {list(value[0].keys())}")
                        
                        if self.rank == 0:
                            print(f"\nServer response structure:")
                            print(f"Result keys: {list(result.keys())}")
                            print(f"Data keys: {list(data_results.keys())}")
                            print(f"Predictions keys: {list(predictions.keys())}")
                        
                        # Process the analysis results to modify texts
                        if not analysis_results:
                            raise ValueError("No analysis results returned from RaTEscore server")
                        
                        # Get number of operations per paragraph from meta info
                        num_operations = extractor_num_ops_per_para or 1
                        
                        # NEW IMPLEMENTATION: Process G/2 responses per prompt instead of global half
                        num_texts = len(analysis_results)
                        num_gens_per_prompt = self.sampling_params.n
                        num_prompts = num_texts // num_gens_per_prompt
                        
                        # Process each analysis result to create edited versions
                        modified_texts = gen_texts.copy()  # Start with copies of original texts
                        operation_info_dict = {"total_operations": 0, "per_prompt": {}}
                        change_flag = False

                        if extractor_num_para == -1:
                            # NEW: For each prompt, select G/2 sources to edit and G/2 targets to replace
                            num_sources_per_prompt = max(1, num_gens_per_prompt // 2)  # G/2, at least 1
                            num_targets_per_prompt = num_sources_per_prompt  # Same number
                            
                            if self.rank == 0:
                                print(f"NEW EditGRPO Implementation:")
                                print(f"Processing {num_sources_per_prompt} sources per prompt (G/2 where G={num_gens_per_prompt})")
                                print(f"Creating {num_targets_per_prompt} edited versions per prompt")
                                print(f"Total prompts: {num_prompts}")
                                print(f"Using random seed: {self.seed}")

                            for prompt_idx in range(num_prompts):
                                # Get indices for this prompt
                                prompt_start = prompt_idx * num_gens_per_prompt
                                prompt_end = prompt_start + num_gens_per_prompt
                                prompt_indices = list(range(prompt_start, prompt_end))
                                
                                # Randomly select source indices (responses to edit)
                                self.paragraph_rng.shuffle(prompt_indices)
                                source_indices = prompt_indices[:num_sources_per_prompt]
                                remaining_indices = prompt_indices[num_sources_per_prompt:]
                                
                                # Randomly select target indices (responses to replace with edited versions)
                                target_indices = remaining_indices[:num_targets_per_prompt]
                                
                                if self.rank == 0 and prompt_idx < 3:  # Log first 3 prompts for debugging
                                    print(f"  Prompt {prompt_idx}: sources={source_indices}, targets={target_indices}")
                                
                                # Create edited versions of source texts
                                for src_idx, tgt_idx in zip(source_indices, target_indices):
                                    if src_idx < len(gen_texts) and tgt_idx < len(analysis_results):
                                        gen_text = gen_texts[src_idx]  # Source text to edit
                                        analysis_result = analysis_results[src_idx]  # Analysis for source
                                        
                                        # Process the analysis result to create edited version
                                        edited_text, text_changed, ops_log = process_text_with_server_analysis(
                                            gen_text, 
                                            analysis_result,
                                            num_operations=num_operations,
                                            similarity_threshold=extractor_similarity_threshold
                                        )
                                        
                                        # Clean any EOS tokens that might have been injected from ground truth
                                        if text_changed:
                                            original_edited = edited_text
                                            edited_text = self._clean_eos_from_text(edited_text)
                                            if self.rank == 0 and original_edited != edited_text:
                                                print(f"DEBUG: Cleaned EOS tokens from RaTEscore server response for prompt {prompt_idx}")
                                        
                                        # Place edited version at target index (not source index!)
                                        if text_changed:
                                            modified_texts[tgt_idx] = edited_text
                                            change_flag = True
                                            
                                            # Update operation info dictionary
                                            prompt_key = str(prompt_idx)
                                            if prompt_key not in operation_info_dict["per_prompt"]:
                                                operation_info_dict["per_prompt"][prompt_key] = {
                                                    "total_ops_for_prompt": 0,
                                                    "edits_summary": []
                                                }
                                            
                                            current_prompt_summary = operation_info_dict["per_prompt"][prompt_key]
                                            num_ops = len(ops_log)
                                            
                                            current_prompt_summary["edits_summary"].append({
                                                "source_index": src_idx,
                                                "target_index": tgt_idx,
                                                "count": num_ops,
                                                "operations": ops_log
                                            })
                                            
                                            current_prompt_summary["total_ops_for_prompt"] += num_ops
                                            operation_info_dict["total_operations"] += num_ops
                        else:
                            # OLD IMPLEMENTATION: Randomly sample specified number of texts using the dedicated RNG
                            available_indices = list(range(num_texts))
                            num_to_sample = min(
                                extractor_num_para or 1,  # Default to 1 if not specified
                                num_texts  # Can't sample more than available
                            )
                            indices_to_process = self.paragraph_rng.sample(
                                available_indices,
                                num_to_sample
                            )
                            
                            if self.rank == 0:
                                print(f"OLD Implementation: Processing {len(indices_to_process)} randomly selected texts: {indices_to_process}")

                            # Only process the selected texts (old way - modify in place)
                            for i in indices_to_process:
                                gen_text = gen_texts[i]
                                analysis_result = analysis_results[i]
                                
                                # Process the analysis result
                                modified_text, text_changed, ops_log = process_text_with_server_analysis(
                                    gen_text, 
                                    analysis_result,
                                    num_operations=num_operations,
                                    similarity_threshold=extractor_similarity_threshold
                                )
                                
                                # Clean any EOS tokens that might have been injected from ground truth
                                if text_changed:
                                    original_modified = modified_text
                                    modified_text = self._clean_eos_from_text(modified_text)
                                    if self.rank == 0 and original_modified != modified_text:
                                        print(f"DEBUG: Cleaned EOS tokens from RaTEscore server response for text {i}")
                                
                                modified_texts[i] = modified_text
                                
                                if text_changed:
                                    change_flag = True
                                    # Calculate original prompt and candidate indices
                                    prompt_idx = i // num_gens_per_prompt
                                    candidate_idx = i % num_gens_per_prompt
                                    prompt_key = str(prompt_idx)
                                    
                                    # Update operation info dictionary
                                    if prompt_key not in operation_info_dict["per_prompt"]:
                                        operation_info_dict["per_prompt"][prompt_key] = {
                                            "total_ops_for_prompt": 0,
                                            "candidates_summary": []
                                        }
                                    
                                    current_prompt_summary = operation_info_dict["per_prompt"][prompt_key]
                                    num_ops = len(ops_log)
                                    
                                    current_prompt_summary["candidates_summary"].append({
                                        "candidate_original_index": candidate_idx,
                                        "count": num_ops,
                                        "operations": ops_log
                                    })
                                    
                                    current_prompt_summary["total_ops_for_prompt"] += num_ops
                                    operation_info_dict["total_operations"] += num_ops
                    else:
                        if self.rank == 0:
                            print(f"Error from RaTEscore extractor server: {response.status_code}, {response.text}")
                        change_flag = False
                        modified_texts = gen_texts
                        operation_info_dict = {"total_operations": 0}
                    
                    if change_flag and self.rank == 0:
                        print("\n=== Text Modification Results (RaTEscore) ===")
                        print(f"Total changes performed: {operation_info_dict['total_operations']}")
                        for prompt_idx_str, prompt_details in operation_info_dict.get('per_prompt', {}).items():
                            print(f"\nPrompt {prompt_idx_str} ({prompt_details['total_ops_for_prompt']} operations):")
                            
                            # Handle new EditGRPO implementation (edits_summary)
                            if 'edits_summary' in prompt_details:
                                for edit_summary in prompt_details.get('edits_summary', []):
                                    src_idx = edit_summary['source_index']
                                    tgt_idx = edit_summary['target_index']
                                    print(f"  Edit: source[{src_idx}] → target[{tgt_idx}] ({edit_summary['count']} operations):")
                                    for op_info in edit_summary.get('operations', []):
                                        print(f"    {op_info['operation']}: {op_info['details']}")
                            
                            # Handle old implementation (candidates_summary)
                            elif 'candidates_summary' in prompt_details:
                                for cand_summary in prompt_details.get('candidates_summary', []):
                                    print(f"  Candidate {cand_summary['candidate_original_index']} ({cand_summary['count']} operations):")
                                    for op_info in cand_summary.get('operations', []):
                                        print(f"    {op_info['operation']}: {op_info['details']}")
                        print("===========================================\n")
                elif extractor_type == "radgraph":
                    # Store original texts for verification
                    original_texts = gen_texts.copy()
                    # Store original response_ids for verification
                    original_response_ids = response_ids.clone()
                    
                    # Call RadGraph extractor server
                    headers = {'Content-Type': 'application/json'}
                    response = requests.post(extractor_server or "http://localhost:5000/extract", json=data, headers=headers)
                    
                    if response.status_code == 200:
                        result = response.json()
                        data = result.get("data", {})
                        predictions = data.get("predictions", {})
                        gen_annotations = predictions.get("gen_annotations", {})
                        gt_annotations = predictions.get("gt_annotations", {})
                        
                        # If no gt_annotations in the expected format, print debug info
                        if self.rank == 0:
                            print(f"\nServer response structure:")
                            print(f"Result keys: {list(result.keys())}")
                            print(f"Data keys: {list(data.keys())}")
                            print(f"Predictions keys: {list(predictions.keys())}")
                            
                        # Handle both formats - either a dictionary of annotations or a single annotation
                        if gt_annotations and 'text' in gt_annotations:
                            # It's already a single annotation object
                            single_gt_annotation = gt_annotations
                            # Convert to the format expected by check_text_field
                            gt_annotations = {"0": single_gt_annotation}
                        
                        # Check if text field is empty in annotations
                        def check_text_field(annotations):
                            if annotations is None:
                                return False
                            if not annotations:  # Empty dictionary
                                return False
                            
                            # If it's a single annotation object (not a dict of annotations)
                            if 'text' in annotations:
                                return annotations.get('text') is not None and annotations.get('text') != ""
                            
                            # Check text field for each entry in the dictionary
                            for key in annotations:
                                if not annotations[key]:
                                    return False
                                text = annotations[key].get('text')
                                if text is None or text == "":
                                    return False
                            return True

                        if not check_text_field(gen_annotations):
                            raise ValueError("Generated text is empty or None in one or more annotations")
                        if not check_text_field(gt_annotations):
                            # Add detailed debugging information
                            print(f"\nGround truth annotations debug:")
                            print(f"Type of gt_annotations: {type(gt_annotations)}")
                            print(f"Content of gt_annotations: {gt_annotations}")
                            
                            # Print the raw response from the server
                            print(f"\nServer response debug:")
                            print(f"Result content: {result}")
                            print(f"Data content: {data}")
                            print(f"Predictions content: {predictions}")
                            
                            # If gt_annotations is a dict, check each key
                            if isinstance(gt_annotations, dict):
                                print("\nGT annotation keys:")
                                for key in gt_annotations:
                                    print(f"Key: {key}, Value type: {type(gt_annotations[key])}")
                                    print(f"Value content: {gt_annotations[key]}")
                            
                            raise ValueError(f"Ground truth text is empty or None in one or more annotations. GT annotations: {gt_annotations}")

                        # (3) Process annotations and modify the text
                        num_operations = extractor_num_ops_per_para or 1
                        num_paragraphs = len(gen_annotations)
                        
                        # Determine which paragraphs to process based on index_paragraph
                        num_gen_paragraphs = len(gen_annotations)
                        if extractor_num_para == -1:
                            # Process a half of the paragraphs
                            index_paragraph = [-1]
                        else:
                            # Randomly sample paragraphs using the dedicated RNG
                            num_to_sample = min(
                                extractor_num_para or 1,  # Default to 1 if not specified
                                num_paragraphs  # Can't sample more than available
                            )
                            available_indices = list(range(num_paragraphs))
                            index_paragraph = self.paragraph_rng.sample(
                                available_indices,
                                num_to_sample
                            )
                        
                        if self.rank == 0:
                            print(f"Processing annotations with {num_operations} operations per paragraph")
                            print(f"Number of paragraphs available: {num_paragraphs}")
                            print(f"Using random seed: {self.seed}")
                            if extractor_num_para == -1:
                                print("Processing a half of the paragraphs")
                            else:
                                print(f"Processing {len(index_paragraph)} randomly selected paragraphs: {index_paragraph}")

                        # from ....utils.radgraph_extractor import process_annotations
                        
                        print("Processing annotations with RadGraph")
                        modified_texts, change_flag, operation_info_dict = process_annotations(
                            gen_annotations, 
                            gt_annotations, 
                            num_operations=num_operations,
                            index_paragraph=index_paragraph
                        )
                        
                        # Clean any EOS tokens that might have been injected from ground truth
                        if change_flag:
                            eos_cleaned_count = 0
                            for i in range(len(modified_texts)):
                                if modified_texts[i] != original_texts[i]:  # Only clean if text was modified
                                    original_modified = modified_texts[i]
                                    modified_texts[i] = self._clean_eos_from_text(modified_texts[i])
                                    if original_modified != modified_texts[i]:
                                        eos_cleaned_count += 1
                            
                            if self.rank == 0 and eos_cleaned_count > 0:
                                print(f"DEBUG: Cleaned EOS tokens from {eos_cleaned_count} RadGraph server responses")
                        
                        if change_flag and self.rank == 0:
                            print("\n=== Text Modification Results (RadGraph) ===")
                            print(f"Total changes performed: {operation_info_dict['total_operations']}")
                            for text_id, details in operation_info_dict['per_paragraph'].items():
                                print(f"\nParagraph {text_id} ({details['count']} operations):")
                                for op_info in details['operations']:
                                    print(f"  {op_info['operation']}: {op_info['details']}")
                            print("=================================\n")
                    else:
                        if self.rank == 0:
                            print(f"Error from RadGraph extractor server: {response.status_code}, {response.text}")
                        change_flag = False
                        modified_texts = gen_texts
                        operation_info_dict = {"total_operations": 0}
                else:
                    # No valid extractor type specified
                    if self.rank == 0:
                        print(f"Warning: Unknown extractor_type '{extractor_type}'. No extraction performed.")
                    change_flag = False
                    modified_texts = gen_texts
                    operation_info_dict = {"total_operations": 0}
                        
                # Common code for both extractors: replace the original response_ids with the modified text tokens
                # should we add "if self.rank == 0" here?
                if change_flag:
                    # Create a mapping of which indices were actually modified
                    modified_indices = []
                    modified_response_ids = []

                    # Track which texts were actually changed
                    for i, (original, modified) in enumerate(zip(original_texts, modified_texts)):
                        if original != modified:
                            modified_indices.append(i)
                            
                            # Final safety check: ensure no EOS tokens remain in the modified text
                            final_cleaned = self._clean_eos_from_text(modified)
                            if final_cleaned != modified and self.rank == 0:
                                print(f"WARNING: Found EOS tokens in final modified text {i} that should have been cleaned earlier")
                                modified = final_cleaned
                            
                            # Only tokenize texts that actually changed
                            tokens = self._clean_text_and_tokenize(modified, self.config.response_length)
                            
                            padded_tokens = torch.full(
                                (self.config.response_length,),
                                self.pad_token_id,
                                dtype=response_ids.dtype,
                                device=response_ids.device
                            )
                            padded_tokens[:len(tokens)] = torch.tensor(
                                tokens, dtype=response_ids.dtype, device=response_ids.device
                            )
                            modified_response_ids.append(padded_tokens)

                    # Replace only the specific indices that were modified
                    for idx, tensor in zip(modified_indices, modified_response_ids):
                        if idx < len(response_ids):
                            response_ids[idx] = tensor
                    
                    # Verification logic - only run on rank 0 for logging
                    if self.rank == 0:
                        print(f"Successfully replaced {len(modified_indices)} responses with modified text")
                        
                        # Verify that modifications were properly applied
                        print("\n=== Text Modification Verification ===")
                        total_operations = operation_info_dict.get('total_operations', 0)
                        print(f"Total operations performed: {total_operations}")
                        print(f"Number of texts modified: {len(modified_indices)}")
                        
                        verification_results = []
                        for i, (original_text, modified_text) in enumerate(zip(original_texts, modified_texts)):
                            # Check if the texts are different
                            is_modified = original_text != modified_text
                            
                            # Check if token IDs were actually changed
                            ids_modified = False
                            if i < len(response_ids) and i < len(original_response_ids):
                                ids_modified = not torch.equal(original_response_ids[i], response_ids[i])
                            
                            # Count operations for this text - handle different structures
                            operations_count = 0
                            if extractor_type == "radgraph":
                                text_id = str(i)
                                operations_count = operation_info_dict.get('per_paragraph', {}).get(text_id, {}).get('count', 0)
                            elif extractor_type == "ratescore":
                                # For RaTEscore, need to map flat index back to prompt/candidate
                                prompt_idx = i // num_gens_per_prompt
                                candidate_idx = i % num_gens_per_prompt
                                prompt_key = str(prompt_idx)
                                
                                # Find operations for this prompt/candidate
                                prompt_info = operation_info_dict.get('per_prompt', {}).get(prompt_key, {})
                                
                                # Handle new EditGRPO implementation (edits_summary)
                                if 'edits_summary' in prompt_info:
                                    for edit_summary in prompt_info.get('edits_summary', []):
                                        if edit_summary.get('target_index') == i:  # This index was a target for editing
                                            operations_count = edit_summary.get('count', 0)
                                            break
                                
                                # Handle old implementation (candidates_summary)
                                elif 'candidates_summary' in prompt_info:
                                    for cand_summary in prompt_info.get('candidates_summary', []):
                                        if cand_summary.get('candidate_original_index') == candidate_idx:
                                            operations_count = cand_summary.get('count', 0)
                                            break
                            
                            verification_results.append({
                                "index": i,
                                "text_modified": is_modified,
                                "ids_modified": ids_modified,
                                "consistent": is_modified == ids_modified,
                                "operations_performed": operations_count,
                                "in_modified_indices": i in modified_indices
                            })

                        # Count statistics
                        total_modified = sum(1 for r in verification_results if r["text_modified"])
                        successfully_modified = sum(1 for r in verification_results if r["consistent"] and r["text_modified"])
                        indices_modified = sum(1 for r in verification_results if r["in_modified_indices"])
                        total_operations_verified = sum(r["operations_performed"] for r in verification_results)

                        print(f"Texts identified as modified: {total_modified}/{len(verification_results)}")
                        print(f"Texts processed for replacement: {indices_modified}/{len(verification_results)}")
                        print(f"Successfully applied: {successfully_modified}/{total_modified}")
                        print(f"Total operations performed: {total_operations_verified}")

                        # List any inconsistencies 
                        inconsistent = [r for r in verification_results if not r["consistent"]]
                        if inconsistent:
                            print("\nInconsistent modifications:")
                            for r in inconsistent:
                                print(f"  - Sample {r['index']}: Text modified={r['text_modified']}, IDs modified={r['ids_modified']}, In modified list={r['in_modified_indices']}")
                        
                        print("=======================================\n")

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(
                        non_tensor_batch["multi_modal_inputs"], self.sampling_params.n
                    )

        # Clean up inappropriate EOS tokens in response_ids BEFORE using it for other tensors
        if self.tokenizer.eos_token_id is not None:
            # First, detect EOS tokens for debugging
            original_eos_count = (response_ids == self.tokenizer.eos_token_id).sum().item()
            
            if original_eos_count > 0:
                if self.rank == 0:
                    print(f"DEBUG: Found {original_eos_count} EOS tokens in response_ids before cleanup")
                
                # Clean up each sequence
                cleaned_sequences = 0
                total_eos_removed = 0
                
                for i in range(response_ids.size(0)):
                    seq = response_ids[i]
                    
                    # DEBUG: Verify memory sharing
                    # '''
                    # seq = response_ids[i]        # seq is a VIEW of response_ids[i]
                    # seq[eos_pos] = self.pad_token_id  # This modifies response_ids[i] directly!

                    # # This is equivalent to:
                    # # response_ids[i][eos_pos] = self.pad_token_id
                    # '''
                    # if self.rank == 0 and i == 0:  # Only log for first sequence on rank 0
                    #     print(f"DEBUG: seq shares memory with response_ids[{i}]: {seq.data_ptr() == response_ids[i].data_ptr()}")
                    
                    # Find positions of EOS tokens
                    eos_mask = (seq == self.tokenizer.eos_token_id)
                    if eos_mask.any():
                        # Find the last non-pad token position
                        non_pad_mask = (seq != self.pad_token_id)
                        if non_pad_mask.any():
                            last_content_pos = non_pad_mask.nonzero(as_tuple=True)[0][-1].item()
                            
                            # Find EOS positions
                            eos_positions = eos_mask.nonzero(as_tuple=True)[0]
                            
                            # Remove EOS tokens that are not at the very end
                            inappropriate_eos = []
                            for eos_pos in eos_positions:
                                eos_pos_item = eos_pos.item()
                                # Keep EOS only if it's at the last content position or right after it
                                if eos_pos_item < last_content_pos:
                                    inappropriate_eos.append(eos_pos_item)
                                elif eos_pos_item == last_content_pos:
                                    # EOS at the last content position - this is acceptable
                                    pass
                                elif eos_pos_item == last_content_pos + 1:
                                    # EOS right after the last content - also acceptable
                                    pass
                                else:
                                    # EOS in padding area - remove it
                                    inappropriate_eos.append(eos_pos_item)
                            
                            # Replace inappropriate EOS tokens with pad tokens
                            if inappropriate_eos:
                                for eos_pos in inappropriate_eos:
                                    seq[eos_pos] = self.pad_token_id
                                cleaned_sequences += 1
                                total_eos_removed += len(inappropriate_eos)
                
                # Report cleanup results
                final_eos_count = (response_ids == self.tokenizer.eos_token_id).sum().item()
                if self.rank == 0:
                    print(f"DEBUG: EOS cleanup completed - Cleaned {cleaned_sequences} sequences")
                    print(f"DEBUG: Removed {total_eos_removed} inappropriate EOS tokens")
                    print(f"DEBUG: EOS count after cleanup: {final_eos_count}")
                    
                    # Show some examples of remaining EOS tokens for verification
                    if final_eos_count > 0:
                        remaining_eos_sequences = []
                        for i, seq in enumerate(response_ids[:10]):  # Check first 10 sequences
                            if (seq == self.tokenizer.eos_token_id).any():
                                eos_positions = (seq == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0].tolist()
                                non_pad_positions = (seq != self.pad_token_id).nonzero(as_tuple=True)[0]
                                last_content_pos = non_pad_positions[-1].item() if len(non_pad_positions) > 0 else -1
                                
                                remaining_eos_sequences.append({
                                    "sequence_idx": i,
                                    "eos_positions": eos_positions,
                                    "last_content_pos": last_content_pos,
                                    "is_appropriate": all(pos >= last_content_pos for pos in eos_positions)
                                })
                        
                        if remaining_eos_sequences:
                            print(f"DEBUG: Remaining EOS tokens (showing first 3):")
                            for seq_info in remaining_eos_sequences[:3]:
                                status = "APPROPRIATE" if seq_info["is_appropriate"] else "INAPPROPRIATE"
                                print(f"  Seq {seq_info['sequence_idx']}: EOS at {seq_info['eos_positions']}, last_content at {seq_info['last_content_pos']} - {status}")
            else:
                if self.rank == 0:
                    print("DEBUG: No EOS tokens found in response_ids")

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_eos_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # Remove the ground_truth from non_tensor_batch after using it
        if "ground_truth" in non_tensor_batch:
            non_tensor_batch.pop("ground_truth")

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
