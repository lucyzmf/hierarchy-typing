# %%
import logging
import typing as tp

import numpy as np
from exca import MapInfra
from scr.experiments.word_feature_decode import WordFeatureDecode
from scr.utils.helper import get_subscore_groups
from scr.utils.subscore_func import seq_onset_offset
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class WordFeatureDecodeSeqOnset(WordFeatureDecode):
    """
    feature decoding timelocked at onset of words or sentences
    """

    subscore_func: str = "null"
    n_relative_pos: int = 2  # how many indexes to test for the next or past
    timelock_onset: bool = True
    sent_onset: bool = False

    infra: MapInfra = MapInfra(version="2")

    def get_data(self, s: str) -> tuple:
        # get events
        events = self.prepare_events()
        subject_df = events[events.subject == s]
        logger.info(f"processing {s}")

        # get feature df
        sel = (subject_df.type == self.feature_class.event_types) & (
            subject_df.is_image == self.is_image
        )

        if not self.onset:
            subject_df.loc[subject_df[sel].index, "start"] = (
                subject_df[sel]["start"] + subject_df[sel]["duration"]
            )

        # load data
        data_dict = self.load_data(subject_df, sel)

        # load
        X = data_dict["neuro"]
        y = data_dict["feature"].numpy()

        # get group for grouped kfold
        group = subject_df[sel][self.group_by].values

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
            subscore_idx = get_subscore_groups(
                subject_df,
                sel,
                self.subscore_func,
                self.feature_class.name,
                self.onset,
            )
        else:
            if "subscore" in self.pipeline.decode_function_name:
                raise ValueError(
                    "using subscore decoding function yet no subscore group provided"
                )
            subscore_idx = None

        if self._apply_mask:
            mask = np.squeeze((y != -1))
            logger.info(f"% of samples {mask.sum()/len(mask)} selected")

            logger.info(np.unique(y[mask], return_counts=True))

            X, group, y = (
                X[mask],
                group[mask],
                y[mask],
            )

        test_epoch_sel = seq_onset_offset(
            subject_df,
            sel,
            self.timelock_onset,
            self.feature_class.event_types,
            self.sent_onset,
        )

        return X, group, y, test_epoch_sel, subscore_idx, subject_df[sel]

    @infra.apply(item_uid=str)
    def run_subjs(self, subj_list: tp.List[str]) -> tp.Iterator[np.ndarray]:
        # build
        n_relative_pos = (
            self.n_relative_pos if not self.timelock_onset else -self.n_relative_pos
        )  # deal with the fact that neg shift is for onset

        for s in subj_list:
            X, group, y, test_epoch_sel, subscore_idx, feature_df = self.get_data(s)
            feature_df = feature_df.reset_index(drop=True)

            feat_type = feature_df.type.unique()[0]

            for (seq_id, timeline), df_ in feature_df.groupby(
                ["sequence_id", "timeline"]
            ):
                if feat_type == "Syllable":
                    feature_df.loc[df_.index, "seq_subword_id"] = np.arange(len(df_))
                    feature_df.loc[df_.index, "seq_subword_id_offset"] = np.flip(
                        np.arange(len(df_))
                    )

            # set shift idx
            if n_relative_pos < 0:  # relative to onset
                shift_size = np.flip(
                    np.arange(n_relative_pos, 1)
                )  # shift -1 gives the next label, hence actual index diff = 1
            else:
                shift_size = np.flip(np.arange(0, n_relative_pos + 1))
            for (seq_id, timeline), df_ in feature_df.groupby(
                ["sequence_id", "timeline"]
            ):
                for s, shift in enumerate(shift_size):
                    feature_df.loc[df_.index, f"shifted_idx{shift}"] = np.roll(
                        df_.index, shift
                    )

            # create data tuples
            datas = [
                (
                    X,
                    y,
                    test_epoch_sel,
                    n_relative_pos,
                    train_idx,
                    test_idx,
                    self.pipeline._pipeline,
                    self.pipeline._eval_function,
                    subscore_idx,
                    feature_df,
                )
                # GroupKFold breaks ties between equal-sized groups with numpy's default sort,
                # so folds differ between CPUs with and without AVX-512.
                for train_idx, test_idx in GroupKFold(n_splits=self.n_splits).split(
                    X, y, group
                )
            ]

            scores = []
            for data in tqdm(datas, desc="Processing folds", unit="fold"):
                scores.append(self.pipeline._decode_function(data))

            scores = np.array(scores)  # n fold n_t

            yield scores
