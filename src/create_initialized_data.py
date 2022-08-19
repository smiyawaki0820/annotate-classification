# -*- coding: utf-8 -*-
# 初めのアノテーション対象を選択

import argparse
import gzip
import json
from typing import List

import numpy as np
import pandas as pd

from spreadsheet_manager import SheetManager


def main(args):
    manager = SheetManager(args.spreadsheet_key)
    sheet_list = manager._list()

    if f"{args.sheet_prefix}-00" in sheet_list:
        raise ValueError(f"sheet_name '{args.sheet_prefix}-00' is already existed!")

    # ソースデータをロード
    sheet = manager._get(f"prepro")
    df = manager._load(sheet)

    # ルールにマッチした行を選択
    df_match = df[df["text"].str.contains(args.rules)]

    # アノテーション対象を書き込むためのシートを作成
    sheet = manager._create(f"{args.sheet_prefix}-00")
    manager._update(sheet, df_match)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create spreadsheet")
    parser.add_argument("--spreadsheet_key", required=True, type=str, 
                          help="'spreadsheet_key' i.e. https://docs.google.com/spreadsheets/d/<spreadsheet_key>/edit")
    parser.add_argument("--sheet_prefix", default="loop", type=str, help="prefix of sheet name: creating ... {prefix}-00")
    parser.add_argument("--rules", default="よい|良い|悪い", type=str, help="tokens for word matching separated by '|'")
    args = parser.parse_args()
    main(args)
