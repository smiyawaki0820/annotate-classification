# -*- coding: utf-8 -*-
import argparse
import gzip
import json

import numpy as np
import pandas as pd

from spreadsheet_manager import SheetManager
from text_normalizer import TextNormalizer


def load_dataframe(path: str, col_text: str, col_label: str) -> pd.DataFrame:
    open_fn = gzip.open if path.endswith(".gz") else open
    assert col_text, "Assertion Error: Please specify the columns for ['text']"
    if path.endswith((".csv", ".csv.gz")):
        df = pd.read_csv(open_fn(path, "rt"))
    elif path.endswith((".tsv", ".tsv.gz")):
        df = pd.read_csv(open_fn(path, "rt"), sep="\t")
    elif path.endswith((".jsonl", ".jsonl.gz")):
        df = pd.read_json(open_fn(path, "rt"), orient='records', lines=True)
    else:
        raise ValueError("Please change your file format that is endswith(('.csv', '.tsv', '.jsonl')) or that of '.gz'")

    df.rename(columns={col_text: "text"}, inplace=True)
    if col_label in df.columns:
        df.rename(columns={col_label: "label"}, inplace=True)
    else:
        df[col_label] = None
    df["index"] = df.index
    col_main = ["index", "label", "text"]
    df = df[col_main + [c for c in df.columns if c not in col_main]]
    return df


def main(args):
    manager = SheetManager(args.spreadsheet_name_or_key)
    sheet_list = manager._list()

    if "source" in sheet_list:
        sheet = manager._get("source")
    else:
        if "シート1" in sheet_list:
            sheet = manager._get("シート1")
            manager._rename(sheet, "source")
        else:
            sheet = manager._create(f"source")

    if args.source_data:
        df = load_dataframe(args.source_data, args.col_text, args.col_label)
        manager._update(sheet, df)

        # 正規化
        normalizer = TextNormalizer()
        df["text"] = df["text"].map(lambda x: normalizer(x))
        if "source" in sheet_list:
            sheet = manager._get("prepro")
        else:
            sheet = manager._create(f"prepro")
        df["label"] = None
        manager._update(sheet, df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create spreadsheet")
    parser.add_argument("--spreadsheet_name_or_key", default="Annotation - sample", type=str, 
                          help="'spreadsheet_name' if not existed else 'spreadsheet_key' i.e. https://docs.google.com/spreadsheets/d/<spreadsheet_key>/edit")
    parser.add_argument("--source_data", default=None, type=str, help="Target data of annotation")
    parser.add_argument("--col_text", default="text", type=str, help="column_name of text in source_data")
    parser.add_argument("--col_label", default="label", type=str, help="column_name of label in source_data")
    args = parser.parse_args()
    main(args)
