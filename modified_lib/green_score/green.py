import re
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
import os
from tqdm import tqdm
import numpy as np
import time
import sys
import warnings
import torch.nn as nn

# Optional vLLM imports
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

# Import necessary functions (ensure these are available in your environment)
from green_score.utils import (
    gather_processes,
    make_prompt,
    clean_responses,
    compute_largest_cluster,
    flatten_values_lists_of_list_dicts_to_dict,
)

def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def tqdm_on_main(*args, **kwargs):
    if is_main_process():
        print("==== Beginning Inference ====")
        return tqdm(*args, **kwargs)
    else:
        return kwargs.get('iterable', None)

class GREEN(nn.Module):
    def __init__(self, model_name, output_dir=".", use_vllm=False, batch_size=None):
        super().__init__()
        warnings.filterwarnings("ignore", message="A decoder-only architecture is being used*")
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning, message="Number of distinct clusters.*")
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

        self.model_name = model_name.split("/")[-1]
        self.output_dir = output_dir
        self.use_vllm = use_vllm
        
        # Check if vLLM is available when requested
        if self.use_vllm and not VLLM_AVAILABLE:
            print("Warning: vLLM requested but not available. Falling back to HuggingFace.")
            self.use_vllm = False
        
        # Set batch size - use provided value or defaults
        if batch_size is not None:
            self.batch_size = batch_size
        else:
            # Default batch sizes if not specified
            self.batch_size = 16 if self.use_vllm else 4
            
        self.max_length = 2048
        
        self.categories = [
            "Clinically Significant Errors",
            "Clinically Insignificant Errors",
            "Matched Findings",
        ]
        self.sub_categories = [
            "(a) False report of a finding in the candidate",
            "(b) Missing a finding present in the reference",
            "(c) Misidentification of a finding's anatomic location/position",
            "(d) Misassessment of the severity of a finding",
            "(e) Mentioning a comparison that isn't in the reference",
            "(f) Omitting a comparison detailing a change from a prior study",
        ]
        self.prompts = None
        self.completions = None
        self.green_scores = None
        self.error_counts = None

        # Distributed setup
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
                torch.cuda.set_device(dist.get_rank())
                if dist.get_rank() == 0:
                    backend_name = "vLLM" if self.use_vllm else "HuggingFace"
                    print(f"Distributed training with {torch.cuda.device_count()} GPUs using {backend_name}")

        # Initialize model based on backend choice
        if self.use_vllm:
            self._init_vllm_model(model_name)
        else:
            self._init_hf_model(model_name)

        # Initialize tokenizer (common for both backends)
        self._init_tokenizer(model_name)

    def _init_vllm_model(self, model_name):
        """Initialize vLLM model"""
        print("Initializing vLLM model for accelerated inference...")
        self.model = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count() if torch.cuda.is_available() and torch.cuda.device_count() > 1 else 1,
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            dtype=torch.float16,
        )
        
        # Set up sampling parameters for deterministic generation
        self.sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic
            max_tokens=2048,
            top_p=1.0,
            stop_token_ids=None,
        )

    def _init_hf_model(self, model_name):
        """Initialize HuggingFace model (original implementation)"""
        print("Initializing HuggingFace model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=False if "Phi" in model_name else True,
            device_map={"": "cuda:{}".format(torch.cuda.current_device())},
            torch_dtype=torch.float16,
        )
        self.model.eval()

    def _init_tokenizer(self, model_name):
        """Initialize tokenizer (common for both backends)"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            add_eos_token=True,
            use_fast=True,
            trust_remote_code=True,
            padding_side="left",
        )

        chat_template = "{% for message in messages %}\n{% if message['from'] == 'human' %}\n{{ '<|user|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'system' %}\n{{ '<|system|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'gpt' %}\n{{ '<|assistant|>\n'  + message['value'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

        self.tokenizer.chat_template = chat_template
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.clean_up_tokenization_spaces = True
        self.tokenizer.padding_side = "left"

    def __call__(self, refs, hyps):
        print("Processing data...making prompts")
        dataset = Dataset.from_dict({"reference": refs, "prediction": hyps})

        dataset = self.process_data(dataset)
        print("Done.")

        self.dataset = dataset

        t = time.time()

        mean, std, green_scores, summary, results_df = self.infer()

        t = time.time() - t
        print("Seconds per example: ", t / len(refs))

        if not is_main_process():
            print(f"Rank {dist.get_rank()} exiting.")
            dist.destroy_process_group()
            sys.exit()

        return mean, std, green_scores, summary, results_df

    def process_data(self, dataset):
        def prompting(examples):
            return {
                "prompt": [
                    make_prompt(r, p)
                    for r, p in zip(examples["reference"], examples["prediction"])
                ]
            }
        dataset = dataset.map(prompting, batched=True)
        return dataset

    @torch.inference_mode()
    def infer(self):
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            dataset_dist = split_dataset_by_node(
                self.dataset,
                rank=get_rank(),
                world_size=int(os.environ["WORLD_SIZE"]),
            )
            print("Distributed dataset created on rank: ", int(os.environ["RANK"]))
        else:
            dataset_dist = self.dataset

        local_completions = []
        local_references = []

        for batch in tqdm_on_main(
            iterable=dataset_dist.iter(batch_size=self.batch_size),
            total=len(dataset_dist) // self.batch_size,
        ):
            local_references.extend(batch["prompt"])
            local_completions.extend(self.get_response(batch))

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.completions, self.prompts = gather_processes(
                local_completions, local_references
            )
        else:
            self.completions = local_completions
            self.prompts = local_references

        if is_main_process():
            print("==== End Inference ====")

        if len(self.completions) != len(self.prompts):
            print("Length of prompts and completions are not equal!")

        return self.process_results()

    def get_response(self, batch):
        """Get response using either vLLM or HuggingFace backend"""
        assert "prompt" in batch.keys(), "prompt is not in batch keys"

        if self.use_vllm:
            return self._get_response_vllm(batch)
        else:
            return self._get_response_hf(batch)

    def _get_response_vllm(self, batch):
        """Get response using vLLM backend"""
        # Prepare prompts for vLLM
        prompts = []
        for prompt in batch["prompt"]:
            chat = [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            prompts.append(formatted_prompt)

        # Use vLLM for generation
        outputs = self.model.generate(prompts, self.sampling_params)

        # Extract generated text
        response_list = []
        for output in outputs:
            response = output.outputs[0].text
            response = clean_responses(response)
            response_list.append(response)

        return response_list

    def _get_response_hf(self, batch):
        """Get response using original HuggingFace backend"""
        batch_prompts = [[{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}] for prompt in batch["prompt"]]

        batch_tokens = self.tokenize_batch_as_chat(batch_prompts)

        outputs = self.model.generate(
            input_ids=batch_tokens["input_ids"],
            attention_mask=batch_tokens["attention_mask"],
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=2048,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        response_list = []
        if isinstance(responses, list):
            for response in responses:
                response = clean_responses(response)
                response_list.append(response)
        else:
            responses = clean_responses(responses)
            response_list.append(responses)

        return response_list

    def tokenize_batch_as_chat(self, batch):
        """Tokenize batch for HuggingFace backend (only used when not using vLLM)"""
        batch = [
            self.tokenizer.apply_chat_template(
                i, tokenize=False, add_generation_prompt=True
            )
            for i in batch
        ]

        batch = self.tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(int(os.environ.get("LOCAL_RANK", 0)))

        return batch

    def process_results(self):
        self.green_scores = [
            self.compute_green(response) for response in self.completions
        ]
        self.error_counts = pd.DataFrame(
            [self.compute_error_count(response) for response in self.completions],
            columns=self.sub_categories + ["Matched Findings"],
        )

        results_df = pd.DataFrame(
            {
                "reference": self.dataset["reference"],
                "predictions": self.dataset["prediction"],
                "evaluation": self.completions,
                "green": self.green_scores,
                **self.error_counts,
            }
        )

        mean, std, summary = self.compute_summary()

        return mean, std, self.green_scores, summary, results_df

    def compute_error_count(self, response):
        _, sig_errors = self.parse_error_counts(response, self.categories[0])
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])
        return sig_errors + [matched_findings]

    def compute_green(self, response):
        sig_present, sig_errors = self.parse_error_counts(response, self.categories[0])
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])

        if matched_findings == 0:
            return 0

        if sig_present is None or matched_findings is None:
            return None

        return matched_findings / (matched_findings + sum(sig_errors))

    def parse_error_counts(self, text, category, for_reward=False):
        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )

        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, text, re.DOTALL)

        sum_counts = 0
        sub_counts = [0 for i in range(6)]

        if not category_text:
            if for_reward:
                return None, None
            return sum_counts, sub_counts
        if category_text.group(1).startswith("No"):
            return sum_counts, sub_counts

        if category == "Matched Findings":
            counts = re.findall(r"^\b\d+\b(?=\.)", category_text.group(1))
            if len(counts) > 0:
                sum_counts = int(counts[0])
            return sum_counts, sub_counts
        else:
            sub_categories = [s.split(" ", 1)[0] + " " for s in self.sub_categories]
            matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

            if len(matches) == 0:
                matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
                sub_categories = [
                    f"({i})" + " " for i in range(1, len(self.sub_categories) + 1)
                ]

            for position, sub_category in enumerate(sub_categories):
                for match in range(len(matches)):
                    if matches[match].startswith(sub_category):
                        count = re.findall(r"(?<=: )\b\d+\b(?=\.)", matches[match])
                        if len(count) > 0:
                            sub_counts[position] = int(count[0])
            return sum(sub_counts), sub_counts

    def parse_error_sentences(self, response, category):
        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )
        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, response, re.DOTALL)
        sub_category_dict_sentences = {}
        for sub_category in self.sub_categories:
            sub_category_dict_sentences[sub_category] = []

        if not category_text:
            return sub_category_dict_sentences
        if category_text.group(1).startswith("No"):
            return sub_category_dict_sentences

        if category == "Matched Findings":
            return (
                category_text.group(1).rsplit(":", 1)[-1].rsplit(".", 1)[-1].split(";")
            )

        matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

        if len(matches) == 0:
            matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
            self.sub_categories = [
                f"({i})" + " " for i in range(1, len(self.sub_categories) + 1)
            ]

        for position, sub_category in enumerate(self.sub_categories):
            for match in range(len(matches)):
                if matches[match].startswith(sub_category):
                    sentences_list = (
                        matches[match].rsplit(":", 1)[-1].split(".", 1)[-1].split(";")
                    )
                    sub_category_dict_sentences[self.sub_categories[position]] = (
                        sentences_list
                    )

        return sub_category_dict_sentences

    def compute_sentences(self, response):
        return self.parse_error_sentences(response, self.categories[0])

    def get_representative_sentences(self, responses):
        list_sentences = []
        for i in responses:
            sentences = self.compute_sentences(i)
            list_sentences.append(sentences)

        dict_sentences = flatten_values_lists_of_list_dicts_to_dict(list_sentences)

        result_sentences_dict = {}

        for i in self.sub_categories:
            sentences = dict_sentences[i]
            sentences = [i for i in sentences if i.strip() != ""]
            _, sentences_of_largest_cluster = compute_largest_cluster(sentences)
            result_sentences_dict[i] = sentences_of_largest_cluster

        return result_sentences_dict

    def compute_accuracy(self, responses):
        counts = []
        for response in responses:
            _, sig_errors = self.parse_error_counts(response, self.categories[0])
            counts.append(sig_errors)

        counts = np.array(counts)

        dict_acc = {}
        for i in range(len(self.sub_categories)):
            error_counts = counts[:, i]
            accuracy = np.mean(error_counts == 0)
            dict_acc[self.sub_categories[i]] = accuracy

        return dict_acc

    def compute_summary(self):
        print("Computing summary ...")
        representative_sentences = self.get_representative_sentences(self.completions)
        accuracies = self.compute_accuracy(self.completions)
        mean = np.mean(self.green_scores)
        std = np.std(self.green_scores)

        summary = f"\n-------------{self.model_name}----------------\n [Summary]: Green average {mean} and standard deviation {std} \n [Clinically Significant Errors Analyses]: <accuracy>. <representative error>\n\n"
        for idx, sub_category in enumerate(self.sub_categories):
            accuracy = accuracies[sub_category]
            sentences = representative_sentences[sub_category]
            summary += f"{sub_category}: {accuracy}. \n {sentences} \n\n"
        summary += "----------------------------------\n"

        return mean, std, summary

if __name__ == "__main__":
    refs = [
        "Interstitial opacities without changes.",
        "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
        "Lung volumes are low, causing bronchovascular crowding. The cardiomediastinal silhouette is unremarkable. No focal consolidation, pleural effusion, or pneumothorax detected. Within the limitations of chest radiography, osseous structures are unremarkable.",
    ]
    hyps = [
        "Interstitial opacities at bases without changes.",
        "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
        "Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis. Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are unchanged.",
    ]

    model_name = "StanfordAIMI/GREEN-radllama2-7b"

    green_scorer = GREEN(model_name, output_dir=".", use_vllm=False)
    mean, std, green_score_list, summary, result_df = green_scorer(refs, hyps)
    print(green_score_list)
    print(summary)
    for index, row in result_df.iterrows():
        print(f"Row {index}:\n")
        for col_name in result_df.columns:
            print(f"{col_name}: {row[col_name]}\n")
        print('-' * 80)

