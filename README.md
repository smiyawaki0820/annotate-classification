# 文書分類用のアノテーション

## 準備

```bash
! bash scripts/setup.sh
```

```python
# スプレッドシートの権限承認
from google.auth import default
from google.colab import auth

import gspread

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
```

## データの前処理 + スプレッドシートの作成

- `--source_data` には、テキストとラベルを持つデータセット `[.tsv, .csv, .jsonl] or .gz` のパスを指定
- テキスト・ラベルフィールドのカラム名を `--col_text, --col_label` で指定

```bash
python src/open_spreadsheet.py \
    --spreadsheet_name_or_key "Annotation - sample" \
    --source_data "dataset.jsonl" \
    --col_text "text" \
    --col_label "label"
```

## 単語マッチによる初期アノテーション対象を指定

- 対象となる単語を `|` 区切りで指定する

```bash
python src/create_initialized_data.py \
    --spreadsheet_key <シートID> \
    --rules "良い|いい|よい|悪い|わるい"
```

## 能動学習によるループ実行

- 以降は、①以下を実行 ②スプレッドシート上でアノテーション、を繰り返す
- `--labels` には対象となるラベルを `,` 区切りで指定する

```bash
python src/annotation.py \
    --spreadsheet_key <シートID> \
    --labels "ニュートラル,ポジティブ,ネガティブ"
```
