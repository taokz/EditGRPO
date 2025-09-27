import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from collections import OrderedDict
from transformers import BertModel, AutoModel, AutoConfig, BertTokenizer
from typing import List, Dict, Union, Optional
from huggingface_hub import hf_hub_download
from appdirs import user_cache_dir

CACHE_DIR = user_cache_dir("chexbert")

def download_model(repo_id, cache_dir, filename=None):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    try:
        hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir, force_filename=filename)
    except Exception as e:
        print(e)

def generate_attention_masks(batch, source_lengths, device):
    """Generate masks for padded batches to avoid self-attention over pad tokens"""
    masks = torch.ones(batch.size(0), batch.size(1), dtype=torch.float)
    for idx, src_len in enumerate(source_lengths):
        masks[idx, src_len:] = 0
    return masks.to(device)

def tokenize(impressions, tokenizer):
    imp = impressions.str.strip()
    imp = imp.replace('\n', ' ', regex=True)
    imp = imp.replace('\s+', ' ', regex=True)
    impressions = imp.str.strip()
    new_impressions = []
    for i in (range(impressions.shape[0])):
        tokenized_imp = tokenizer.tokenize(impressions.iloc[i])
        if tokenized_imp:  # not an empty report
            res = tokenizer.encode_plus(tokenized_imp)['input_ids']
            if len(res) > 512:  # length exceeds maximum size
                res = res[:511] + [tokenizer.sep_token_id]
            new_impressions.append(res)
        else:  # an empty report
            new_impressions.append([tokenizer.cls_token_id, tokenizer.sep_token_id])
    return new_impressions

class bert_labeler(nn.Module):
    def __init__(self, p=0.1, clinical=False, freeze_embeddings=False, pretrain_path=None, inference=False, **kwargs):
        """ Init the labeler module
        @param p (float): p to use for dropout in the linear heads, 0.1 by default is consistant with
                          transformers.BertForSequenceClassification
        @param clinical (boolean): True if Bio_Clinical BERT desired, False otherwise. Ignored if
                                   pretrain_path is not None
        @param freeze_embeddings (boolean): true to freeze bert embeddings during training
        @param pretrain_path (string): path to load checkpoint from
        """
        super(bert_labeler, self).__init__()

        if pretrain_path is not None:
            self.bert = BertModel.from_pretrained(pretrain_path)
        elif clinical:
            self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        elif inference:
            config = AutoConfig.from_pretrained('bert-base-uncased')
            self.bert = AutoModel.from_config(config)
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p)
        # size of the output of transformer's last layer
        hidden_size = self.bert.pooler.dense.in_features
        # classes: present, absent, unknown, blank for 12 conditions + support devices
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
        # classes: yes, no for the 'no finding' observation
        self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

    def forward(self, source_padded, attention_mask):
        """ Forward pass of the labeler
        @param source_padded (torch.LongTensor): Tensor of word indices with padding, shape (batch_size, max_len)
        @param attention_mask (torch.Tensor): Mask to avoid attention on padding tokens, shape (batch_size, max_len)
        @returns out (List[torch.Tensor])): A list of size 14 containing tensors. The first 13 have shape
                                            (batch_size, 4) and the last has shape (batch_size, 2)
        """
        # shape (batch_size, max_len, hidden_size)
        final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]
        # shape (batch_size, hidden_size)
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        cls_hidden = self.dropout(cls_hidden)
        out = []
        for i in range(14):
            out.append(self.linear_heads[i](cls_hidden))
        return out

class WeightedCheXbert(nn.Module):
    """
    A class for calculating weighted classwise accuracy for CheXbert predictions.
    Takes the same input format as F1CheXbert (text reports) and outputs weighted accuracy metrics.
    """
    
    def __init__(self, refs_filename=None, hyps_filename=None, device=None, **kwargs):
        super(WeightedCheXbert, self).__init__()
        self.refs_filename = refs_filename
        self.hyps_filename = hyps_filename

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device(device)

        # Model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = bert_labeler(inference=True)

        # Download pretrained CheXbert model
        checkpoint = os.path.join(CACHE_DIR, "chexbert.pth")
        if not os.path.exists(checkpoint):
            download_model(repo_id='StanfordAIMI/RRG_scorers', cache_dir=CACHE_DIR, filename="chexbert.pth")

        # Load model
        state_dict = torch.load(checkpoint, map_location=self.device)['model_state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v

        # Load params
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        # Define the 14 target classes as in F1CheXbert
        self.target_names = [
            "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
            "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
            "Fracture", "Support Devices", "No Finding"
        ]

    def get_label(self, report, mode="rrg"):
        """Convert text report to binary labels using CheXbert"""
        impressions = pd.Series([report])
        out = tokenize(impressions, self.tokenizer)
        batch = torch.LongTensor([o for o in out])
        src_len = [b.shape[0] for b in batch]
        attn_mask = generate_attention_masks(batch, src_len, self.device)
        out = self.model(batch.to(self.device), attn_mask)
        out = [out[j].argmax(dim=1).item() for j in range(len(out))]
        v = []
        if mode == "rrg":
            for c in out:
                if c == 0: # blank class is NaN
                    v.append('')
                if c == 3: # x uncertain class is 1 (as positive); <---> uncertain as negative (0)
                    v.append(1)
                    # v.append(0)
                if c == 2: # negative class is 0 
                    v.append(0)
                if c == 1: # posotive class is 1
                    v.append(1)
            v = [1 if (isinstance(l, int) and l > 0) else 0 for l in v]
        else:
            raise NotImplementedError(mode)
        return v

    def calculate_weighted_accuracy(self, y_true_np, y_pred_np, class_frequencies=None):
        """Calculate weighted accuracy metrics for all classes using proper inverse frequency weighting"""
        if class_frequencies is None:
            class_frequencies = {}
            for i, class_name in enumerate(self.target_names):
                class_frequencies[class_name] = float(np.mean(y_true_np[:, i] == 1))
        
        results = {"per_class": {}, "overall": {}}
        overall_weighted_acc = 0
        total_weight = 0
        
        for i, class_name in enumerate(self.target_names):
            class_true = y_true_np[:, i]
            class_pred = y_pred_np[:, i]
            
            pos_samples = (class_true == 1)
            neg_samples = (class_true == 0)
            
            # Get class frequency (num of a class / total num)
            freq = class_frequencies.get(class_name, float(np.mean(pos_samples)))
            
            # Calculate inverse frequency weights
            pos_weight = 1/freq if freq > 0 else 0
            neg_weight = 1/(1-freq) if freq < 1 else 0
            
            # Normalize weights to sum to 1
            total = pos_weight + neg_weight
            if total > 0:
                pos_weight = pos_weight / total
                neg_weight = neg_weight / total
            
            # Calculate accuracies
            pos_correct = np.sum((class_pred == 1) & pos_samples)
            pos_total = np.sum(pos_samples)
            pos_acc = float(pos_correct / pos_total if pos_total > 0 else 0)
            
            neg_correct = np.sum((class_pred == 0) & neg_samples)
            neg_total = np.sum(neg_samples)
            neg_acc = float(neg_correct / neg_total if neg_total > 0 else 0)
            
            # Calculate weighted accuracy for this class
            # weighted_acc = (pos_acc * pos_weight + neg_acc * neg_weight)
            weighted_acc = (pos_correct * pos_weight + neg_correct * neg_weight) / 2
            # weighted_acc = (pos_acc + neg_acc) / 2
            
            results["per_class"][class_name] = {
                "positive_accuracy": pos_acc,
                "negative_accuracy": neg_acc,
                "weighted_accuracy": weighted_acc,
                "positive_weight": pos_weight,
                "negative_weight": neg_weight,
                "frequency": freq
            }
            
            overall_weighted_acc += weighted_acc
            total_weight += 1
        
        results["overall"]["weighted_accuracy"] = float(overall_weighted_acc)
        results["overall"]["mean_weighted_accuracy"] = float(overall_weighted_acc / total_weight)
        return results

    def forward(self, hyps, refs, class_frequencies=None):
        """
        Process text reports and calculate weighted accuracy metrics.
        
        Args:
            hyps: List of hypothesis (predicted) text reports
            refs: List of reference (ground truth) text reports
            class_frequencies: Optional dict of class frequencies for weighting
            
        Returns:
            dict: Dictionary containing weighted accuracy metrics
        """
        # Convert text reports to labels
        if self.refs_filename is None:
            refs_labels = [self.get_label(l.strip()) for l in refs]
        else:
            if os.path.exists(self.refs_filename):
                refs_labels = [eval(l.strip()) for l in open(self.refs_filename).readlines()]
            else:
                refs_labels = [self.get_label(l.strip()) for l in refs]
                open(self.refs_filename, 'w').write('\n'.join(map(str, refs_labels)))

        hyps_labels = [self.get_label(l.strip()) for l in hyps]
        if self.hyps_filename is not None:
            open(self.hyps_filename, 'w').write('\n'.join(map(str, hyps_labels)))

        # Convert to numpy arrays
        refs_np = np.array([np.array(r) for r in refs_labels])
        hyps_np = np.array([np.array(h) for h in hyps_labels])

        # Calculate weighted accuracy metrics
        results = self.calculate_weighted_accuracy(refs_np, hyps_np, class_frequencies)
        
        return results
