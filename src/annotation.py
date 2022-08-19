# -*- coding: utf-8 -*-
# SMALL-TEXT を使用して能動学習を行う

import argparse
import gzip
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import datasets
from transformers import BertJapaneseTokenizer

from small_text.active_learner import PoolBasedActiveLearner
from small_text.query_strategies import LeastConfidence
from small_text.integrations.transformers import TransformerModelArguments
from small_text.integrations.transformers.classifiers.factories import TransformerBasedClassificationFactory
from small_text.integrations.transformers.datasets import TransformersDataset

from spreadsheet_manager import SheetManager
from trainer import TransformersWrapper
from create_dataset import MyDataset


RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
END = "\033[0m"



def main(args):
    manager = SheetManager(args.spreadsheet_key)
    sheet_list = manager._list()

    sheet_prepro = manager._get("prepro")
    df_data = manager._load(sheet_prepro).set_index("index")

    print(GREEN + "|--> 学習データをロード" + END)
    sheet_vmax = 0
    for title in sheet_list:
        if title.startswith(args.sheet_prefix):
            sheet_vmax = max(int(title.split("-")[-1]), sheet_vmax)
            sheet = manager._get(title)
            df = manager._load(sheet).set_index("index")
            df_data["label"].loc[df.index] = df["label"]

    df_train = df_data[~(df_data["label"] == "")]
    df_test = df_data[df_data["label"] == ""]

    n_annot = df_train.shape[0]
    n_data = df_data.shape[0]
    ratio_annot = 100 * n_annot / n_data
    print(f"# of annotations ::: {ratio_annot:.2f}% ({n_annot}/{n_data})")
    distrib_label = df_data["label"].value_counts().to_dict()
    print("label distributions ::: ", distrib_label)

    print(GREEN + "|--> アノテーション済データを 'prepro' シートに統合" + END)
    manager._update(sheet_prepro, df_data.sort_index().reset_index())

    print(GREEN + "|--> 学習" + END)
    trainer = TransformersWrapper(args, df_train.reset_index())
    trainer.train()

    print(GREEN + "|--> 次のアノテーション対象となるインデックスを取得" + END)
    # indices = df_data[df_data["label"] == ""].sample(100).index.values
    df_test.reset_index(inplace=True)
    query_strategy = LeastConfidence()
    test_data = MyDataset(df_test, trainer.label2id, trainer.tokenizer, is_test=True).test_data
    indices = query_strategy.query(
        trainer,
        test_data,
        indices_unlabeled=df_test.index.values,
        indices_labeled=np.array([]),
        y=None,
        n=100
    )
    df_target = df_data.loc[df_test.iloc[indices]["index"]]

    print(GREEN + "|--> 新たにシートを作成" + END)
    sheet = manager._create(f"{args.sheet_prefix}-{sheet_vmax+1:02}")
    manager._update(sheet, df_target.reset_index())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create spreadsheet")
    parser.add_argument("--spreadsheet_key", required=True, type=str, 
                          help="'spreadsheet_key' i.e. https://docs.google.com/spreadsheets/d/<spreadsheet_key>/edit")
    parser.add_argument("--sheet_prefix", default="loop", type=str, help="prefix of sheet name: creating ... {prefix}-00")
    parser = TransformersWrapper.add_parser(parser)
    args = parser.parse_args()
    main(args)
