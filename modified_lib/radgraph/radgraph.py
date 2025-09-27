import logging
import torch.nn as nn
import numpy as np

import json
import os
import tarfile
import torch
import importlib.metadata
from dotmap import DotMap

from radgraph.allennlp.data import Vocabulary
from radgraph.allennlp.data.dataset_readers import AllennlpDataset
from radgraph.allennlp.data.dataloader import PyTorchDataLoader
from radgraph.allennlp.data import token_indexers
from radgraph.allennlp.modules import token_embedders, text_field_embedders
from radgraph.allennlp.common.params import Params
from radgraph.dygie.data.dataset_readers.dygie import DyGIEReader
from radgraph.dygie.models import dygie

from radgraph.utils import download_model
from radgraph.utils import (
    preprocess_reports,
    postprocess_reports,
    batch_to_device,
)

from radgraph.rewards import compute_reward
from appdirs import user_cache_dir

logging.getLogger("radgraph").setLevel(logging.CRITICAL)
logging.getLogger("allennlp").setLevel(logging.CRITICAL)

MODEL_MAPPING = {
    "radgraph": "radgraph.tar.gz",
    "radgraph-xl": "radgraph-xl.tar.gz",
}

version = importlib.metadata.version('radgraph')
CACHE_DIR = user_cache_dir("radgraph")
CACHE_DIR = os.path.join(CACHE_DIR, version)


class RadGraph(nn.Module):
    def __init__(
            self,
            batch_size=1,
            cuda=None,
            model_type=None,
            **kwargs
    ):

        super().__init__()
        # Device handling. For now we stick to cpu.
        if cuda is None:
            cuda = 0 if torch.cuda.is_available() else -1

        # Fix: Respect the cuda parameter when setting device
        if cuda == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type is None:
            print("model_type not provided, defaulting to radgraph-xl")
            model_type = "radgraph-xl"

        assert model_type in ["radgraph", "radgraph-xl"]

        self.cuda = cuda
        self.batch_size = batch_size
        self.model_type = model_type.lower()
        self.model_path = os.path.join(CACHE_DIR, MODEL_MAPPING[model_type])
        temp_dir = os.path.join(CACHE_DIR, model_type)

        try:
            if not os.path.exists(self.model_path):
                download_model(
                    repo_id="StanfordAIMI/RRG_scorers",
                    cache_dir=CACHE_DIR,
                    filename=MODEL_MAPPING[model_type],
                )
        except Exception as e:
            print("Model download error", e)

        # Archive
        if not os.path.exists(temp_dir):
            with tarfile.open(self.model_path, "r:gz") as tar:
                tar.extractall(path=temp_dir)

        # Read config.
        config_path = os.path.join(temp_dir, "config.json")
        config = json.load(open(config_path))
        config = DotMap(config)

        # Vocab
        vocab_dir = os.path.join(temp_dir, "vocabulary")
        vocab_params = config.get("vocabulary", Params({}))
        vocab = Vocabulary.from_files(
            vocab_dir, vocab_params.get("padding_token"), vocab_params.get("oov_token")
        )

        # Tokenizer
        tok_indexers = {
            "bert": token_indexers.PretrainedTransformerMismatchedIndexer(
                model_name=config.dataset_reader.token_indexers.bert.model_name,
                max_length=config.dataset_reader.token_indexers.bert.max_length
            )
        }
        self.reader = DyGIEReader(max_span_width=config.dataset_reader.max_span_width, token_indexers=tok_indexers)

        # Create embedder
        token_embedder = token_embedders.PretrainedTransformerMismatchedEmbedder(
            model_name=config.model.embedder.token_embedders.bert.model_name,
            max_length=config.model.embedder.token_embedders.bert.max_length
        )
        embedder = text_field_embedders.BasicTextFieldEmbedder({"bert": token_embedder})

        # Model
        model_dict = config.model
        for name in ["type", "embedder", "initializer", "module_initializer"]:
            del model_dict[name]

        model = dygie.DyGIE(vocab=vocab,
                            embedder=embedder,
                            **model_dict
                            )

        model_state_path = os.path.join(temp_dir, "weights.th")
        # Fix: Always use map_location when cuda=-1 to ensure CPU loading
        if cuda == -1:
            model_state = torch.load(model_state_path, map_location=torch.device('cpu'))
        else:
            if device.type == "cpu":
                model_state = torch.load(model_state_path, map_location=torch.device('cpu'))
            else:
                model_state = torch.load(model_state_path)

        model.load_state_dict(model_state, strict=True)
        model.eval()

        self.device = device
        self.model = model.to(device)
        
        # Add debug print to confirm device assignment
        print(f"RadGraph initialized with device: {self.device}")

    def forward(self, hyps):

        assert isinstance(hyps, str) or isinstance(hyps, list)
        if isinstance(hyps, str):
            hyps = [hyps]

        hyps = ["None" if not s else s for s in hyps]

        # Preprocessing
        model_input = preprocess_reports(hyps, self.model_type)

        instances = [self.reader.text_to_instance(line) for line in model_input]
        data = AllennlpDataset(instances)
        data.index_with(self.model.vocab)
        iterator = PyTorchDataLoader(batch_size=1, dataset=data)

        # Forward
        results = []
        for batch in iterator:
            batch = batch_to_device(batch, self.device)
            output_dict = self.model(**batch)
            results.append(self.model.make_output_human_readable(output_dict).to_json())

        # Postprocessing
        inference_dict = postprocess_reports(results)
        return inference_dict


class F1RadGraph(nn.Module):
    def __init__(
            self,
            reward_level,
            model_type=None,
            **kwargs
    ):

        super().__init__()
        assert reward_level in ["simple", "partial", "complete", "all"]
        self.reward_level = reward_level
        self.radgraph = RadGraph(model_type=model_type, **kwargs)

    def forward(self, refs, hyps):
        # Checks
        assert isinstance(hyps, str) or isinstance(hyps, list)
        assert isinstance(refs, str) or isinstance(refs, list)

        if isinstance(hyps, str):
            hyps = [hyps]
        if isinstance(refs, str):
            refs = [refs]

        assert len(refs) == len(hyps)

        # getting empty report list
        number_of_reports = len(hyps)
        empty_report_index_list = [i for i in range(number_of_reports) if (len(hyps[i]) == 0) or (len(refs[i]) == 0)]
        number_of_non_empty_reports = number_of_reports - len(empty_report_index_list)

        # stacking all reports (hyps and refs)
        report_list = [
                          hypothesis_report
                          for i, hypothesis_report in enumerate(hyps)
                          if i not in empty_report_index_list
                      ] + [
                          reference_report
                          for i, reference_report in enumerate(refs)
                          if i not in empty_report_index_list
                      ]

        assert len(report_list) == 2 * number_of_non_empty_reports

        # getting annotations
        inference_dict = self.radgraph(report_list)

        # Compute reward
        reward_list = []
        hypothesis_annotation_lists = []
        reference_annotation_lists = []
        non_empty_report_index = 0
        for report_index in range(number_of_reports):
            if report_index in empty_report_index_list:
                if self.reward_level == "all":
                    reward_list.append(((0., 0., 0.), (0., 0., 0.), (0., 0., 0.)))
                else:
                    reward_list.append(0.)
                continue

            hypothesis_annotation_list = inference_dict[str(non_empty_report_index)]
            reference_annotation_list = inference_dict[
                str(non_empty_report_index + number_of_non_empty_reports)
            ]

            reward = compute_reward(
                hypothesis_annotation_list,
                reference_annotation_list,
                self.reward_level,
            )
            reward_list.append(reward)
            reference_annotation_lists.append(reference_annotation_list)
            hypothesis_annotation_lists.append(hypothesis_annotation_list)
            non_empty_report_index += 1

        assert non_empty_report_index == number_of_non_empty_reports

        if self.reward_level == "all":
            # Process reward_list, ensuring each sub-element in r is a tuple of length 3
            processed_rewards = [
                tuple(
                    (0.0, 0.0, 0.0) if not isinstance(sub_r, tuple) or len(sub_r) != 3 else sub_r
                    for sub_r in r
                ) if isinstance(r, tuple) else ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
                for r in reward_list
            ]

            try:
                # Unpack the processed reward_list
                simple_precision, simple_recall, simple_f1 = zip(*[r[0] for r in processed_rewards])
                partial_precision, partial_recall, partial_f1 = zip(*[r[1] for r in processed_rewards])
                complete_precision, complete_recall, complete_f1 = zip(*[r[2] for r in processed_rewards])

                # Compute mean for each metric
                mean_simple_precision = np.mean(simple_precision)
                mean_simple_recall = np.mean(simple_recall)
                mean_simple_f1 = np.mean(simple_f1)

                mean_partial_precision = np.mean(partial_precision)
                mean_partial_recall = np.mean(partial_recall)
                mean_partial_f1 = np.mean(partial_f1)

                mean_complete_precision = np.mean(complete_precision)
                mean_complete_recall = np.mean(complete_recall)
                mean_complete_f1 = np.mean(complete_f1)

                # Return mean metrics as a tuple of tuples
                mean_reward = (
                    (mean_simple_precision, mean_simple_recall, mean_simple_f1),
                    (mean_partial_precision, mean_partial_recall, mean_partial_f1),
                    (mean_complete_precision, mean_complete_recall, mean_complete_f1)
                )
            except Exception as e:
                print(f"Error processing reward_list: {e}")
                print(reward_list)
                mean_reward = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        else:
            mean_reward = np.mean(reward_list)

        return (
            mean_reward,
            reward_list,
            hypothesis_annotation_lists,
            reference_annotation_lists,
        )


if __name__ == "__main__":
    model_type = "radgraph-xl"
    radgraph = RadGraph(model_type=model_type)

    refs = ["no acute cardiopulmonary abnormality",
            "et tube terminates 2 cm above the carina retraction by several centimeters is recommended for more optimal placement bibasilar consolidations better assessed on concurrent chest ct",
            "there is no significant change since the previous exam the feeding tube and nasogastric tube have been removed",
            "unchanged mild pulmonary edema no radiographic evidence pneumonia",
            "no evidence of acute pulmonary process moderately large size hiatal hernia",
            "no acute intrathoracic process"]

    annotations = radgraph(["no evidence of acute cardiopulmonary process moderate hiatal hernia"])
    json_output = "tests/annotations_{}.json".format(model_type)
    if not os.path.exists(json_output):
        json.dump(annotations, open(json_output, "w"))
    else:
        assert annotations == json.load(open(json_output, "r")), annotations
        print("annotations matches")

    hyps = ["no acute cardiopulmonary abnormality",
            "endotracheal tube terminates 2 5 cm above the carina bibasilar opacities likely represent atelectasis or aspiration",
            "there is no significant change since the previous exam",
            "unchanged mild pulmonary edema and moderate cardiomegaly",
            "no evidence of acute cardiopulmonary process moderate hiatal hernia",
            "no acute cardiopulmonary process"]
    del radgraph
    model_type = "radgraph"
    f1radgraph = F1RadGraph(reward_level="all", model_type=model_type)
    mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=hyps,
                                                                                                   refs=refs)
    print(mean_reward)
    print(reward_list)
    print(mean_reward == (0.6238095238095238, 0.5111111111111111, 0.5011204481792717))
    # (0.6238095238095238, 0.5111111111111111, 0.5011204481792717)
    # ([1.0, 0.4, 0.5714285714285715, 0.8, 0.5714285714285715, 0.4],
    #  [1.0, 0.26666666666666666, 0.5714285714285715, 0.4, 0.42857142857142855, 0.4],
    #  [1.0, 0.23529411764705885, 0.5714285714285715, 0.4, 0.4, 0.4])
