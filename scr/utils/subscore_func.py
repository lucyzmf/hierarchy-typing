import numpy as np
import pandas as pd


# select seq onset or offset epochs
def seq_onset_offset(
    df: pd.DataFrame, sel: pd.Series, onset: bool, event_type: str, sent_onset: bool
) -> np.array:
    MAP = {
        "Button": {
            "True": "char_id",
            "False": "char_id_offset",
        },
        "Syllable": {
            "True": "subword_index",
            "False": "subword_index_offset",
        },
        "Word": {
            "True": "word_id",
            "False": "word_id_offset",
        },
    }

    df = df[sel]

    sel_sent_onset = (
        df["word_id"].values == 0 if onset else df["word_id_offset"].values == 0
    )

    if event_type != "Word":
        event_sel = df[MAP[event_type][str(onset)]].values == 0
        if sent_onset:
            event_sel = np.logical_and(event_sel, sel_sent_onset)
        return event_sel
    else:  # word feature
        if sent_onset:
            return (
                df[MAP[event_type][str(onset)]].values == 0
            )  # if sent onset, consider only the first word
        else:
            return np.ones(len(df), dtype=bool)  # all words are considered
