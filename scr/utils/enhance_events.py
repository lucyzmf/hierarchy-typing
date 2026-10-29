import logging
import typing as tp
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd
import pydantic
import spacy
from spacy_syllables import SpacySyllables  # noqa
from tqdm import tqdm

from neuralset.enhancers import BaseEnhancer

from .feature_dicts import segmentation_df_spa, word_accent_match_spa

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def clean_accents(stim: str) -> str:
    """clean accents from a string
    á, é, í, ó, ú, ü, ñ -> a, e, i, o, u, u, n
    """
    if type(stim) == str:
        stim = stim.lower()

        stim = stim.replace("á", "a")
        stim = stim.replace("é", "e")
        stim = stim.replace("í", "i")
        stim = stim.replace("ó", "o")
        stim = stim.replace("ú", "u")
        stim = stim.replace("ü", "u")
        stim = stim.replace("ñ", "n")

    return stim


def accent_lookup(word: str) -> str:
    out = word_accent_match_spa.loc[word, "word"]
    return out


# Helper function for processing each group
def process_group(group_data, feat_func, target_feature, with_accents):
    i, group = group_data

    word = group.text.iloc[0]
    # check language
    if group.language.iloc[0] == "spanish":
        word = accent_lookup(word)
    features = feat_func(word)

    if features is None:
        return None

    group = group.reset_index(drop=True)
    typed = group.button.values
    log_list = []
    count_dropped_features = 0

    for f in range(len(features)):
        feat = features[f]
        feat_no_dash = feat.replace("-", "")

        feat_no_dash_without_accent = clean_accents(feat_no_dash)

        if feat_no_dash_without_accent in "".join(typed):
            typed_str = "".join(typed)
            position = typed_str.index(feat_no_dash_without_accent)
            this_feat = group.iloc[position : position + len(feat_no_dash_without_accent)]
            group = group.drop(
                np.arange(position, position + len(feat_no_dash_without_accent)),
                axis="index",
            ).reset_index(drop=True)
            typed = group.button.values
        else:
            count_dropped_features += 1
            continue

        first_key = this_feat.iloc[0]
        last_key = this_feat.iloc[-1]

        feat_dict = dict(
            type=target_feature,
            text=(
                feat_no_dash if with_accents else feat_no_dash_without_accent
            ),  # in text only keep the letters
            word=word,
            start=first_key.start,
            duration=last_key.start + last_key.duration - first_key.start,
            subword_index=f,
            subword_index_offset=len(features) - 1 - f,
        )

        other_info = this_feat.iloc[0][
            [
                "is_image",
                "word_id",
                "word_index",
                "trial_id",
                "language",
                "study",
                "subject",
                "timeline",
            ]
        ]

        feat_dict.update(other_info)
        log_list.append(feat_dict)

    return log_list, count_dropped_features


def add_event(
    events: pd.DataFrame, target_feature: str, feat_func: callable, with_accents: bool
):
    typed_df = events[
        (events.type == "Button")
        & (~events.button.isin(["<space>", "<special>"]))
        & (~events.word_index.isna())
    ]
    dfg = typed_df.groupby(["subject", "timeline", "word_index"])

    log_list = []
    total_dropped_features = 0

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_group,
                (i, group),
                feat_func,
                target_feature,
                with_accents,
            ): i
            for i, group in dfg
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                logs, dropped = result
                log_list.extend(logs)
                total_dropped_features += dropped

    logger.info(
        f"{total_dropped_features / (len(log_list) + total_dropped_features)} features dropped out of all subjects"
    )

    log_df = pd.DataFrame(log_list)

    log_df = log_df.sort_values(by=["subject", "timeline", "trial_id", "start"])

    if target_feature == "Syllable":
        log_df.text = log_df.text.fillna("nan")

    return pd.concat([events, log_df], ignore_index=True)


class CleanButtons(BaseEnhancer):
    """Clean buttons that are not space or special characters for each timeline"""

    name: tp.Literal["CleanButtons"] = "CleanButtons"
    model_config = pydantic.ConfigDict(extra="forbid")

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        special_char = [
            "<special>",
            "<space>",
            "\x14",
            "<number>",
            "£",
            "¤",
            "¿",
            "`",
            "-",
        ]
        excluded_num = events.button.isin(special_char).sum() / len(events)
        logger.info(f"{excluded_num} special button events removed ")
        events = events[~events.button.isin(special_char)]

        # some button duration is 0 in eeg
        sel = (events.type == "Button") & (events.duration == 0)
        events.loc[sel, "duration"] = 0.05

        # exclude buttons that could not be assigned to a word
        logger.info(
            f"{(events.text=='FAILED').sum()/(events.type=='Button').sum()} FAILED button events removed "
        )
        failed_buttons = events[(events.type == "Button") & (events.text == "FAILED")]
        events = events[~events.index.isin(failed_buttons.index)]

        return events


def _get_syllables(text: str, nlp: spacy.lang) -> tp.List[str]:
    return nlp(text)[0]._.syllables


def subword_lookup_spa(text: str, event_type: str) -> tp.List[str]:
    return segmentation_df_spa.loc[text, event_type]


class AddSyllableEvents(BaseEnhancer):
    """Adds syllable events to the event dataframe"""

    name: tp.Literal["AddSyllableEvents"] = "AddSyllableEvents"
    event_type: tp.Literal["Syllable"] = "Syllable"
    language: tp.Literal["spanish", "english"] = "spanish"
    _feat_func: callable
    model_config = pydantic.ConfigDict(extra="forbid")
    LANGUAGES: tp.ClassVar[tp.Dict[str, str]] = dict(
        english="en_core_web_lg", french="fr_core_news_lg", spanish="es_core_news_lg"
    )
    with_accents: bool = False

    def model_post_init(self, log__: tp.Any) -> None:
        if self.language == "english":
            from neuralset import utils

            model = utils.get_spacy_model(model=self.LANGUAGES[self.language])

            # check if syllables are already in the pipeline
            if "syllables" not in model.pipe_names:
                model.add_pipe("syllables")
            self._feat_func = partial(_get_syllables, nlp=model)
        elif self.language == "spanish":
            self._feat_func = partial(subword_lookup_spa, event_type=self.event_type)
        else:
            raise ValueError(f"language {self.language} not supported")

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        events = add_event(events, self.event_type, self._feat_func, self.with_accents)
        return events


class AddContextPinet(BaseEnhancer):

    name: tp.Literal["AddContextPinet"] = "AddContextPinet"
    model_config = pydantic.ConfigDict(extra="forbid")

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        words = events[(events.type == "Word") & (~events.word_id.isna())]

        def _get_context_by_id(row):
            true_seq = row.true_sequence
            word_list = true_seq.split(" ")
            context = word_list[: int(row.word_id + 1)]
            return " ".join(context)

        # change context column dtype to str
        events["context"] = None
        events.loc[words.index, "context"] = words.apply(_get_context_by_id, axis=1)

        words = events[events.type == "Word"]
        logger.info(
            f"{words.context.isna().sum()/len(words)} words have problematic context generation"
        )

        # remove words with problematic context generation
        events = events.drop(words[words.context.isna()].index)

        return events


class RemoveTrainingTrials(BaseEnhancer):
    "remove the first two sequences of each block as they should not be included in the analysis"

    name: tp.Literal["RemoveTrainingTrials"] = "RemoveTrainingTrials"
    model_config = pydantic.ConfigDict(extra="forbid")

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        events = events[~(events.trial_id.isin([0, 1]))]

        return events


class ReassignSequenceid(BaseEnhancer):
    "reassign sequence id so that each id corresponds to a unique sequence"

    name: tp.Literal["ReassignSequenceid"] = "ReassignSequenceid"
    model_config = pydantic.ConfigDict(extra="forbid")

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        # get all unique sentences from perception
        sentences_prod = events.loc[
            (events["is_image"] == False) & (events["type"] == "Sentence"), "text"
        ].unique()
        sentence_percep = events.loc[
            (events["is_image"] == True) & (events["type"] == "Sentence"), "text"
        ].unique()

        sentences = np.unique(np.concatenate([sentences_prod, sentence_percep]))
        uid = pd.Series(range(len(sentences)), index=sentences)

        # Create a mapping of (timeline, trial_id) to new sequence_id
        mapping = events.loc[
            events["text"].isin(sentences), ["timeline", "trial_id", "text"]
        ]
        mapping = mapping.drop_duplicates()
        mapping["sequence_id"] = mapping["text"].map(uid)
        mapping["map_key"] = (
            mapping["timeline"].astype(str) + "_" + mapping["trial_id"].astype(str)
        )

        events["map_key"] = (
            events["timeline"].astype(str) + "_" + events["trial_id"].astype(str)
        )
        events = events.merge(
            mapping[["map_key", "sequence_id"]], on="map_key", how="left"
        )
        events = events.drop(columns=["map_key"])

        return events


class AddSeqOffsetWordId(BaseEnhancer):
    """add word id relative to seq offset"""

    name: tp.Literal["AddSeqOffsetWordId"] = "AddSeqOffsetWordId"
    model_config = pydantic.ConfigDict(extra="forbid")

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        for i, group in events.groupby(
            ["subject", "timeline", "sequence_id", "is_image"]
        ):
            word_id_max = group.word_id.max()
            events.loc[group.index, "word_id_offset"] = word_id_max - group.word_id
        return events


class AddCharIdOffset(BaseEnhancer):
    """add char id relative to word offset"""

    name: tp.Literal["AddCharIdOffset"] = "AddCharIdOffset"
    model_config = pydantic.ConfigDict(extra="forbid")

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        buttons = events[events.type == "Button"]

        for i, group in buttons.groupby(
            ["subject", "timeline", "sequence_id", "is_image", "word_index"]
        ):

            max_char_id = group.char_id.max()
            char_id_offset = max_char_id - group.char_id
            events.loc[group.index, "char_id_offset"] = char_id_offset

        return events


class AddSeqCharIdOnsetOffset(BaseEnhancer):
    """add char id relative to sequence"""

    name: tp.Literal["AddSeqCharIdOnsetOffset"] = "AddSeqCharIdOnsetOffset"
    model_config = pydantic.ConfigDict(extra="forbid")

    def __call__(self, events: pd.DataFrame) -> pd.DataFrame:
        buttons = events[events.type == "Button"]

        for i, group in buttons.groupby(
            ["subject", "timeline", "sequence_id", "is_image"]
        ):
            events.loc[group.index, "seq_char_id"] = np.arange(len(group))
            events.loc[group.index, "seq_char_id_offset"] = np.flip(np.arange(len(group)))

        return events


