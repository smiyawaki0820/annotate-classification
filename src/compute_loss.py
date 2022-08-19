# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List
from cytoolz import curry

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
import torch.nn as nn

from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


@curry
def compute_metrics(pred, pi_y:torch.tensor=None, tau=0.05):
    labels = pred.label_ids
    preds = torch.tensor(pred.predictions)
    
    # Post-hoc logit adjustment
    preds = preds - tau * torch.log(pi_y.repeat(preds.shape[0], 1)).to(preds.device)
    preds = preds.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def adjusted_cross_entropy(y_pred, y_true, pi_y, tau=0.05):
    bunsi = torch.exp(y_pred + tau * torch.log(pi_y.repeat(y_pred.shape[0], 1)))
    bunbo = torch.sum(torch.exp(y_pred + tau * torch.log(pi_y.repeat(y_pred.shape[0], 1))), dim=-1).unsqueeze(-1)
    softmax = bunsi / bunbo
    negative_likelihood = - torch.log(softmax[range(y_true.shape[0]), y_true])
    return negative_likelihood #.mean()


@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.
    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """
    epsilon: float = 0.1
    ignore_index: int = -100
    pi_y: torch.tensor = None

    def __call__(self, model_output, labels, shift_labels=False):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        # nll_loss = log_probs.gather(dim=-1, index=labels)
        nll_loss = adjusted_cross_entropy(logits, labels.squeeze(), self.pi_y).unsqueeze(-1)

        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)
        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss
