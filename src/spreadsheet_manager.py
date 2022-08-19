# -*- coding: utf-8 -*-

from typing import List

import pandas as pd

from google.auth import default
from google.colab import auth

import gspread
from gspread.models import Spreadsheet, Worksheet


class SheetManager:
    def __init__(self, spreadsheet_name_or_key: str):
        auth.authenticate_user()
        creds, _ = default()
        self.gc = gspread.authorize(creds)
        self.__post_init__(spreadsheet_name_or_key)

    def __post_init__(self, spreadsheet_name_or_key: str):
        try:
            self.spreadsheet = self.gc.open_by_key(spreadsheet_name_or_key)
            print("Open the spreasheet ...")
        except:
            self.spreadsheet = self.gc.create(spreadsheet_name_or_key)
            print("Create a new spreasheet ...")

    @property
    def _url(self,) -> str:
        return self.spreadsheet.url

    def _list(self,) -> List[str]:
        return [sheet.title for sheet in self.spreadsheet.worksheets()]

    def _create(self, sheet_name: str) -> Worksheet:
        return self.spreadsheet.add_worksheet(title=sheet_name, rows=30, cols=10)

    def _get(self, sheet_name: str) -> Worksheet:
        return self.spreadsheet.worksheet(sheet_name)
    
    def _load(self, sheet: Worksheet) -> pd.DataFrame:
        return pd.DataFrame(sheet.get_all_records())

    def _update(self, sheet: Worksheet, df: pd.DataFrame):
        sheet.update([df.columns.values.tolist()] + df.values.tolist())
        print("Updated sheet ::: ", sheet.url)

    def _rename(self, sheet: Worksheet, new_sheet_name: str):
        sheet.update_title(new_sheet_name)

    def _delete(self, sheet: Worksheet):
        self.spreadsheet.del_worksheet(sheet)
