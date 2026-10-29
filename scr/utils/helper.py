import numpy as np
import pandas as pd


def get_subscore_groups(
    subj_df: pd.DataFrame,
    sel: pd.Series,
    subscore_method: str,
    feature_name: str,
    onset: bool,
) -> np.array:
    """get groups for subscore method

    Args:
        subj_df (pd.DataFrame): subject dataframe
        subscore_method (str): subscore method
        feature_name (str): feature name

    Returns:
        np.array: groups
    """
    if subscore_method == "index":
        if "Syllable" in feature_name:
            if onset:
                return subj_df[sel].subword_index.values
            else:
                return subj_df[sel].subword_index_offset.values
        elif "Button" in feature_name:
            if onset:
                return subj_df[sel].char_id.values
            else:
                return subj_df[sel].char_id_offset.values
        else:
            return subj_df[sel].word_id.values
    else:
        raise ValueError(f"subscore method {subscore_method} not found")


# clean df subjects
def clean_subjects(events: pd.DataFrame):
    """take full events dataframe and clean up subject numbers so that events from the same subjects from different days are
    counted as the same subject

    Args:
        events (pd.DataFrame): full events dataframe from pinetmeg
    """

    events.loc[events.subject == "Pinet2024Meg/S18", "subject"] = "Pinet2024Meg/S1"
    events.loc[events.subject == "Pinet2024Meg/S14", "subject"] = "Pinet2024Meg/S4"
    events.loc[events.subject == "Pinet2024Meg/S10", "subject"] = "Pinet2024Meg/S5"
    events.loc[events.subject == "Pinet2024Meg/S21", "subject"] = "Pinet2024Meg/S5"

    return events


