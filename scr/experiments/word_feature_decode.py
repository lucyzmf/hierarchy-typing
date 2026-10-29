# %%
import logging


import typing as tp

import numpy as np
import pandas as pd
import pydantic
import tqdm
from exca import MapInfra
from pydantic import BaseModel
from scr.utils.features import *
from scr.utils.helper import clean_subjects, get_subscore_groups
from scr.utils.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

import neuralset as ns

from ..utils.neuro_features import Neuro_devices

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class WordFeatureDecode(BaseModel):
    """
    decoding of context / word / syllable / key features around word onset or offset
    """

    num_subj: str | int = "all"

    # study params
    study: ns.data.StudyLoader

    # data feature params
    device: Neuro_devices

    is_image: bool = False
    onset: bool = True  # if true at onset, false, plus duration for start time

    # feature
    feature_class: Features
    subscore_func: str = "null"
    group_by: str = "sequence_id"

    # epoch params
    start: float = -0.5
    duration: float = 1.5

    # model and testing
    pipeline: Pipeline
    n_splits: int = 5
    _apply_mask: bool = False

    # misc
    # if true, decode amongst right keys, if false decode amongst left keys
    exclude_left_keys: bool = False

    infra: MapInfra = MapInfra(version="v4")
    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    _exclude_from_cls_uid = (
        "infra",
        "num_subj",
    )

    def model_post_init(self, log__: tp.Any) -> None:
        if "Syllable" in self.feature_class.name:
            assert self.is_image == False, "no is_image=True for subword features"

        # verify
        logger.info(
            f"{self.feature_class.name} taking event type {self.feature_class.event_types}"
        )

        if hasattr(self.feature_class, "need_mask"):
            self._apply_mask = True
            logger.info(f"exclude uncategoriesd features in {self.feature_class.name}")

        logger.info(f"testing {self.feature_class.name}")

    def prepare_events(self) -> pd.DataFrame:
        # Only compute events if not cached
        events = self.study.build()

        if self.exclude_left_keys:
            events = events[~(events.is_left_key == self.exclude_left_keys)]

        events = clean_subjects(events)  # Clean subjects

        return events

    def load_data(self, subj_df, sel):
        segments = ns.segments.list_segments(
            subj_df,
            idx=sel,
            start=self.start,
            duration=self.duration,
        )
        dataset = ns.SegmentDataset(
            features={"neuro": self.device, "feature": self.feature_class},
            segments=segments,
        )

        return dataset.as_one_batch().data

    def prepare_features(self, events: pd.DataFrame):
        # prepare
        self.device.prepare(events)
        self.feature_class.prepare(events)

    def run(self):
        logger.info(f"testing {self.feature_class.name}")
        events = self.prepare_events()
        self.prepare_features(events)

        subjs = events.subject.unique()

        if self.num_subj != "all":
            subjs = subjs[: self.num_subj]

        all_scores = self.run_subjs(subjs)

        return dict(zip(subjs, all_scores))

    def get_data(self, s: str) -> tuple:
        events = self.prepare_events()
        subject_df = events[events.subject == s]

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

        if self.pipeline.model in [
            "RidgeClassifierCV",
        ]:  # classifier models need 1D class labels
            y = np.reshape(y, -1)
            assert len(y.shape) == 1
        elif self.pipeline.model == "RidgeCV":
            # check if for button class
            if self.feature_class.name == "ButtonFeat":
                y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
                logger.info(f"one hot encoded y shape {y.shape}")

        if self._apply_mask:
            mask = np.squeeze((y != -1))
            logger.info(f"% of samples {mask.sum()/len(mask)} selected")

            logger.info(np.unique(y[mask], return_counts=True))

            X, group, y = (
                X[mask],
                group[mask],
                y[mask],
            )

            if subscore_idx is not None:
                subscore_idx = subscore_idx[mask]

        return X, group, y, subscore_idx

    @infra.apply(item_uid=str, exclude_from_cache_uid=("num_subj",))
    def run_subjs(self, subj_list: tp.List[str]) -> tp.Iterator[np.ndarray]:
        # build
        for s in subj_list:
            X, group, y, subscore_idx = self.get_data(s)

            datas = [
                (
                    X,
                    y,
                    subscore_idx,
                    train_idx,
                    test_idx,
                    self.pipeline._pipeline,
                    self.pipeline._eval_function,
                )
                # GroupKFold breaks ties between equal-sized groups with numpy's default sort,
                # so folds differ between CPUs with and without AVX-512.
                for train_idx, test_idx in GroupKFold(n_splits=self.n_splits).split(
                    X, y, group
                )
            ]

            scores = []
            for data in tqdm.tqdm(datas, desc="Processing folds", unit="fold"):
                scores.append(self.pipeline._decode_function(data))

            scores = np.array(scores)  # n fold n_t

            yield scores
