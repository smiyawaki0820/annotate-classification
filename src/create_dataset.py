# -*- coding: utf-8 -*-

import tempfile
import pandas as pd
from sklearn.model_selection import train_test_split
import datasets


class MyDataset:
    def __init__(self,
            df,
            label2id,
            tokenizer,
            num_classes: int = 3,
            is_test: bool = False,
        ):
        self.label2id = label2id
        self.id2label = {v:k for k,v in label2id.items()}
        self.num_classes = num_classes

        self.tokenizer = tokenizer
        self.is_test = is_test
        if not is_test:
            df_train, df_valid = self.train_test_split(df)
            self.train_data = self._dataset(df_train)
            self.valid_data = self._dataset(df_valid)
        else:
            self.test_data = self._dataset(df)

    def encode_fn(self, batch):
        kwargs_encode = {
            "add_special_tokens": True,
            "padding": "max_length",
            "max_length": 128,
            "return_attention_mask": True,
            "return_tensors": "pt",
            "truncation": "longest_first",
        }
        encoded_text = self.tokenizer(batch["text"], **kwargs_encode)
        batch.update(encoded_text)
        batch["label"] = [self.label2id.get(label) for label in batch["label"]]
        return batch

    def _dataset(self, df) -> datasets.Dataset:
        data = datasets.Dataset.from_pandas(df)
        data.set_transform(self.encode_fn, columns=["index", "text", "label"], output_all_columns=True)
        _data = []
        for idx in range(len(data)):
            obj = data[idx]
            _data.append(obj)
        return _data

    @staticmethod
    def train_test_split(df: pd.DataFrame, test_size=0.1, shuffle=True, random_state=2022):
        return train_test_split(
            df,
            test_size=test_size,
            shuffle=shuffle,
            stratify=df["label"],
            random_state=random_state,
        )
