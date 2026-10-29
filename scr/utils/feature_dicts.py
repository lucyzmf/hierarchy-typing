# load correct syllable segmentation
import ast
import os
from pathlib import Path

import pandas as pd

wk_dir = Path(os.path.dirname(os.path.realpath(__file__)))
pinet_dir = wk_dir.parent.parent


word_accent_match_spa = pd.read_csv(f"{pinet_dir}/assets/words_noaccent_match.csv")
word_accent_match_spa.set_index("no_accents", inplace=True)


segmentation_df_path_spa = pinet_dir / "assets" / "subword_segmentation_final.csv"
segmentation_df_spa = pd.read_csv(segmentation_df_path_spa, delimiter=";")
segmentation_df_spa.set_index("words", inplace=True)
segmentation_df_spa["Syllable"] = segmentation_df_spa["Syllable"].apply(ast.literal_eval)

# a script to keep all the feature dicts to keep the feature.py clean
key_to_int = {
    "s": 0,
    "o": 1,
    "t": 2,
    "e": 3,
    "n": 4,
    "c": 5,
    "i": 6,
    "a": 7,
    "r": 8,
    "d": 9,
    "l": 10,
    "b": 11,
    "z": 12,
    "v": 13,
    "f": 14,
    "m": 15,
    "u": 16,
    "h": 17,
    "p": 18,
    "g": 19,
    "q": 20,
    "w": 21,
    "x": 22,
    "y": 23,
    "j": 24,
    "ý": 25,
    "k": 26,
    "ü": 27,
    "û": 28,
    "<space>": 31,
    "<special>": 32,
    "\x14": 33,
    "<number>": 34,
    # french
    "ô": 35,
    "è": 36,
    "ù": 37,
    "é": 38,
    "â": 39,
    "ê": 40,
    "ç": 41,
    "à": 42,
}


