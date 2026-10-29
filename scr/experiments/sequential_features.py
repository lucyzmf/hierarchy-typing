# %%
import logging


import typing as tp

import sklearn
import sklearn.model_selection
import tqdm
from sklearn.preprocessing import OneHotEncoder

from ..utils.features import *
from ..utils.helper import get_subscore_groups
from .word_feature_decode import WordFeatureDecode

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

event_to_idcolumn = {
    "Syllable": "subword_index",
    "Word": "word_id",
    "Button": "char_id",
}


class SequentialDecoding(WordFeatureDecode):
    """
    hierarchical decoding for words, perception or typing
    """

    cv: tp.Literal["StratifiedKFold", "KFold", "GroupKFold", "StratifiedGroupKFold"] = (
        "StratifiedKFold"
    )

    # subscoring
    relative_to_seq: bool = False  # deprecated, kept for config backward compatibility

    # num of shifts
    shifts: int = 1

    infra: MapInfra = MapInfra(version="v4")
    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    _exclude_from_cls_uid = ("infra",)

    def get_data(self, s: str) -> tuple:
        events = self.prepare_events()
        subject_df = events[events.subject == s]
        logger.info(f"processing {s}")

        # perception
        sel = (subject_df.type == self.feature_class.event_types) & (
            subject_df.is_image == self.is_image
        )

        # load data
        data_dict = self.load_data(subject_df, sel)

        # load
        X = data_dict["neuro"]
        y = data_dict["feature"].numpy()

        if (
            self.pipeline.model == "RidgeClassifierCV"
        ):  # take int per sample as class lable
            y = np.reshape(y, -1)
            assert len(y.shape) == 1
        elif self.pipeline.model == "RidgeCV":
            # check if for button class
            if self.feature_class.name == "ButtonFeat":
                y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
                logger.info(f"one hot encoded y shape {y.shape}")

        # retrieve subscore
        if self.subscore_func != "null":
            if "subscore" not in self.pipeline.decode_function_name:
                raise ValueError(
                    "using subscore group yet no subscore decoding function provided"
                )
            logger.info(f"subscoring based on {self.subscore_func}")
            subscore_idx = (
                get_subscore_groups(  # this is the content vs function word subscore
                    subject_df,
                    sel,
                    self.subscore_func,
                    self.feature_class.name,
                    onset=True,
                )
            )
        else:
            if "subscore" in self.pipeline.decode_function_name:
                raise ValueError(
                    "using subscore decoding function yet no subscore group provided"
                )
            subscore_idx = None

        # get feature index
        pos = subject_df[sel][event_to_idcolumn[self.feature_class.event_types]].values

        group = subject_df[sel].sequence_id.values if "GroupKFold" in self.cv else None

        return X, y, pos, subscore_idx, group, subject_df[sel]

    @infra.apply(item_uid=str, exclude_from_cache_uid=("num_subj",))
    def run_subjs(self, subj_list: tp.List[str]) -> tp.Iterator[np.ndarray]:
        # define decoding fun
        cv = getattr(sklearn.model_selection, self.cv)

        for s in subj_list:
            X, y, pos, subscore_idx, group, feature_df = self.get_data(s)
            feature_df = feature_df.reset_index(drop=True)

            feat_type = self.feature_class.event_types

            if feat_type == "Syllable":
                for (seq_id, timeline), df_ in feature_df.groupby(
                    ["sequence_id", "timeline"]
                ):
                    feature_df.loc[df_.index, "seq_subword_id"] = np.arange(len(df_))
                    feature_df.loc[df_.index, "seq_subword_id_offset"] = np.flip(
                        np.arange(len(df_))
                    )

            # set shift idx
            shift_sizes = np.arange(-self.shifts, self.shifts + 1)
            for (seq_id, timeline), df_ in feature_df.groupby(
                ["sequence_id", "timeline"]
            ):
                for s, shift in enumerate(shift_sizes):
                    feature_df.loc[df_.index, f"shifted_idx{shift}"] = np.roll(
                        df_.index, shift
                    )
                    # split arg
            split_arg = (X, y, group) if "GroupKFold" in self.cv else (X, y)

            # create data tuples
            datas = [
                (
                    X,
                    y,
                    pos,
                    self.shifts,
                    train_idx,
                    test_idx,
                    self.pipeline._pipeline,
                    self.pipeline._eval_function,
                    subscore_idx,
                    feature_df,
                )
                # GroupKFold breaks ties between equal-sized groups with numpy's default sort,
                # so folds differ between CPUs with and without AVX-512.
                for train_idx, test_idx in cv(n_splits=self.n_splits).split(*split_arg)
            ]

            scores = []
            for data in tqdm.tqdm(datas, desc="Processing folds", unit="fold"):
                scores.append(self.pipeline._decode_function(data))

            # each entry of scores is one fold
            scores = np.array(scores)

            # transpose dimensions, always the first two dims
            transpose_dims = [1, 0] + list(range(2, len(scores.shape)))
            scores = np.transpose(scores, transpose_dims)

            yield scores
