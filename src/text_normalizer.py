# -*- coding: utf-8 -*-
from collections import OrderedDict
import re
from typing import List
import unicodedata

from bunkai import Bunkai

import spacy
from sudachipy import tokenizer
from sudachipy import dictionary


class TextNormalizer:
    def __init__(self):
        self.bunkai = Bunkai()
        self.tokenizer_obj = dictionary.Dictionary().create()
        self.mode = tokenizer.Tokenizer.SplitMode.C

    def __call__(self, text: str):
        sents = []
        text = text.replace("[SEP]", "")
        for sent in self.bunkai(text):
            candidate = self.normalize(sent)
            if candidate:
                sents.append(candidate.strip())
        return " [SEP] ".join(sents)

    def normalize(self, text: str):
        text = self.nfkc(text)
        text = self.regex(text)
        kana_ratio = self.kana_ratio(text)
        if kana_ratio < 0.1:
            return None

        tokens = []
        for token in self.tokenizer_obj.tokenize(text, self.mode):
            tokens.append({
                "surface": token.surface(),
                "pos": token.part_of_speech()[0],
                "norm": token.normalized_form()
            })

        n_content = sum(token["pos"] in self._content_pos for token in tokens)
        if n_content < 2:
            return None
        
        surfaces = [t["surface"] for t in tokens]
        type_token_ratio = len(set(surfaces)) / (len(surfaces)+0.01)
        if type_token_ratio < 0.5:
            return None

        _norm = lambda token: token["norm"] if token["pos"] in self._normalizing_pos else token["surface"]
        normalized_tokens = [_norm(token) for token in tokens]
        return "".join(normalized_tokens)

    def nfkc(self, text: str) -> str:
        return unicodedata.normalize("NFKC", text)
    
    def regex(self, text: str) -> str:
        for _, (src, tgt) in self._patterns.items():
            text = re.sub(src, tgt, text)
        return text

    def kana_ratio(self, text: str) -> float:
        # ひらがな・カタカナの割合が 30 ％以下 （漢字や英語が多い場合） であれば除去
        n_char = len(text)
        if n_char == 0:
            return 0.0
        return sum(self.is_kana(c) for c in text) / (n_char)

    @staticmethod
    def is_kana(ch:str):
        try:
            return unicodedata.name(ch).startswith(("HIRAGANA LETTER", "KATAKANA LETTER"))
        except ValueError:
            return False

    @property
    def _patterns(self):
        return OrderedDict(
            parethesis = ('[\(|（|\<|【].*?[\)|）|\>|】]', ''),
            symbols = ('["#&\'\\\\()*,-./;<=>@\^_`{|}〔〕“”〈〉『』【】＊（）＃＠｀＋￥・\.]', ''),
        )

    @property
    def _content_pos(self):
        return ("名詞", "代名詞", "形容詞", "動詞", "形状詞", "副詞")

    @property
    def _normalizing_pos(self):
        return ("名詞")


if __name__ == "__main__":
    norm = TextNormalizer()
    print(norm("吾輩（私）はかわいかった、綺麗な猫である(^^)"))
