# -*- coding: utf-8 -*-
import argparse
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import datasets
from transformers import (
    BertConfig, BertJapaneseTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments,
    EarlyStoppingCallback,
)

from spreadsheet_manager import SheetManager
from utils import set_seed, get_module
from compute_loss import LabelSmoother, compute_metrics
from create_dataset import MyDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor, pi_y=self.pi_y)
        super().compute_loss(model, inputs, return_outputs)


class MyPredDataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        inputs = {
            "input_ids": d["input_ids"],
            "attention_mask": d["attention_mask"],
            "token_type_ids": d["token_type_ids"],
        }
        return inputs


class TransformersWrapper:
    @classmethod
    def add_parser(cls, parser):
        parser.add_argument_group("Args for Model")
        parser.add_argument("--config_name_or_path", default="cl-tohoku/bert-base-japanese-v2", type=str)
        parser.add_argument("--config_class", default="transformers.BertConfig", type=str)
        parser.add_argument("--model_name_or_path", default="cl-tohoku/bert-base-japanese-v2", type=str)
        parser.add_argument("--model_class", default="transformers.BertForSequenceClassification", type=str)
        parser.add_argument("--tokenizer_name_or_path", default="cl-tohoku/bert-base-japanese-v2", type=str)
        parser.add_argument("--tokenizer_class", default="transformers.BertJapaneseTokenizer", type=str)
        parser.add_argument("--labels", default="ニュートラル,ポジティブ,ネガティブ", type=str)

        parser.add_argument_group("Args for TrainingArguments")
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--max_steps", default=100, type=int)
        parser.add_argument("--weight_decay", default=0.1, type=float)
        parser.add_argument("--max_grad_norm", default=1.0, type=float)
        parser.add_argument("--per_device_train_batch_size", default=32, type=int)
        parser.add_argument("--per_device_eval_batch_size", default=128, type=int)
        parser.add_argument("--logging_steps", default=50, type=int)
        parser.add_argument("--eval_steps", default=50, type=int)
        parser.add_argument("--save_steps", default=10000, type=int)
        parser.add_argument("--output_dir", default="tmp", type=str)
        parser.add_argument("--not_overwrite_output_dir", action="store_false")
        parser.add_argument("--save_strategy", default="steps", type=str, choices=["steps"])
        parser.add_argument("--evaluation_strategy", default="steps", type=str, choices=["steps"])
        parser.add_argument("--optim", default="adamw_hf", type=str)
        parser.add_argument("--not_dataloader_pin_memory", action="store_false")
        parser.add_argument("--not_load_best_model_at_end", action="store_false")
        parser.add_argument("--seed", default=3407, type=int)
        parser.add_argument("--label_smoothing_factor", default=0.3, type=float)
        parser.add_argument("--metric_for_best_model", default="eval_f1", type=str)
        parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
        return parser

    def __init__(self, args, df):
        labels = args.labels.split(",")
        self.label2id = {label:idx for idx, label in enumerate(labels)}
        self.id2label = {idx:label for idx, label in enumerate(labels)}

        self.config = get_module(args.config_class).from_pretrained(args.config_name_or_path, num_labels=len(labels), id2label=self.id2label)
        self.model = get_module(args.model_class).from_pretrained(args.model_name_or_path, config=self.config)
        self.tokenizer = get_module(args.tokenizer_class).from_pretrained(args.tokenizer_name_or_path)

        dataset = MyDataset(df, self.label2id, self.tokenizer)
        self.train_data = dataset.train_data
        self.valid_data = dataset.valid_data

        counter = df["label"].value_counts().to_dict()
        pi_y = torch.tensor([counter[label] for label in labels], dtype=torch.float).to(device)
        pi_y /= torch.sum(pi_y, dim=-1)
        self.metrics_fn = compute_metrics(pi_y=pi_y)

        self.trainer = self._trainer(args)
        setattr(self.trainer, "pi_y", pi_y)

    def train(self,):
        train_result = self.trainer.train()
        print(train_result)
        train_metrics = train_result.metrics
        self.trainer.log_metrics("train", train_metrics)

    def predict_proba(self, dataset: datasets.Dataset):
        dataloader = DataLoader(MyPredDataset(dataset), batch_size=128, shuffle=False)
        self.model.eval()
        self.model.to(device)
        proba = []
        with torch.no_grad():
            for bix, batch in enumerate(dataloader):
                batch = {k:torch.tensor(v, device=device) if k in ["input_ids", "token_type_ids", "attention_mask"] else v for k, v in batch.items()}
                out = self.model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    token_type_ids=batch["token_type_ids"].to(device),
                    labels=None,
                )
                logits = F.softmax(out.logits, dim=-1)
                if bix == 0:
                    proba = deepcopy(logits)
                else:
                    proba = torch.cat([proba, logits], dim=0)
        return proba.cpu().detach().numpy()

    def _training_args(self, args):
        return TrainingArguments(
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            output_dir=args.output_dir,
            overwrite_output_dir=args.not_overwrite_output_dir,
            save_strategy=args.save_strategy,
            evaluation_strategy=args.evaluation_strategy,
            optim=args.optim,
            dataloader_pin_memory=args.not_dataloader_pin_memory,
            load_best_model_at_end=args.not_load_best_model_at_end,
            seed=args.seed,
            label_smoothing_factor=args.label_smoothing_factor,
            metric_for_best_model=args.metric_for_best_model,
        )

    def _trainer(self, args):
        return Trainer(
            model=self.model,
            args=self._training_args(args),
            train_dataset=self.train_data,
            eval_dataset=self.valid_data,
            compute_metrics = self.metrics_fn,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=20)],
        )
