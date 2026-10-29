# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import typing as tp

import mne
import mne.io.brainvision
import numpy as np
import pandas as pd
import pydantic
import sklearn.preprocessing
import torch

import neuralset as ns
from neuralset.infra import MapInfra

from .base import BaseDynamic

Raw: tp.TypeAlias = mne.io.Raw | mne.io.brainvision.brainvision.RawBrainVision
logger = logging.getLogger(__name__)


class Meg(BaseDynamic):
    """If frequency is set to "native", the frequency used will be the one provided by the Meg event
    filter and resample preprocessing steps can be cached.

    Parameters
    ----------
    baseline :
        If provided as a tuple (start, end), corresponds to the start and end times (in seconds)
        relative to the **beginning of a window** (i.e. NOT relative to the epoch onset as opposed
        to MNE's convention) of the segment to use for baselining.
    """

    name: tp.Literal["Meg"] = "Meg"
    event_types: tp.Literal["Meg", "Eeg", "Emg", "Fnirs", "Ieeg"] = "Meg"

    frequency: tp.Literal["native"] | float = "native"
    offset: float = 0.0
    baseline: tp.Tuple[float, float] | None = None
    pick_types: tp.Tuple[str, ...] = pydantic.Field(("meg",), min_length=1)
    apply_proj: bool = False
    filter: tp.Tuple[float | None, float | None] | None = None
    apply_hilbert: bool = False
    notch_filter: float | None = None
    mne_cpus: int = -1
    infra: MapInfra = MapInfra(
        timeout_min=120,
        gpus_per_node=0,
        cpus_per_task=10,
        version="1",
    )
    scaler: None | tp.Literal["RobustScaler", "StandardScaler"] = None
    clamp: float | None = None
    find_bad_channels: tp.Optional[tp.Literal["lof", "maxwell"]] = None

    _channels: tp.Dict[str, int] = {}

    @classmethod
    def _exclude_from_cls_uid(cls) -> tp.List[str]:
        prev = super()._exclude_from_cls_uid()
        return prev + ["mne_cpus"]

    def _exclude_from_cache_uid(self) -> tp.List[str]:
        prev = super()._exclude_from_cache_uid()
        return prev + ["baseline", "offset", "clamp"]

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        # Update channel mapping to be robust to mne.Raw
        self._channels = {}
        # check baseline
        if self.baseline is not None:
            message = f"baseline must be None or 2 floats, got {self.baseline}"
            assert len(self.baseline) == 2, message
            assert isinstance(self.baseline[0], float), message
            assert isinstance(self.baseline[1], float), message
            assert self.baseline[1] > self.baseline[0], message

    def prepare(self, events: pd.DataFrame) -> None:
        """specify how to load and preprocess the event.
        Can be overriden by user.
        """
        events_ = self._events_from_dataframe(events)
        self._get_preprocessed_data(events_)
        self._prepare_channels(events_)

    # NOTE: We use the FIF format to cache MEG as we don't want to discard information such
    # as projectors and channel info which cannot be saved in the faster BrainVision format.
    # However, FIF files take a lot longer to read - if this becomes a bottleneck, we might need to
    # look into using another file format for MEG too.
    @infra.apply(
        item_uid=lambda e: str(e.filepath),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
        cache_type="MneRawFif",
    )
    def _get_preprocessed_data(
        self, events: tp.List[ns.events.Meg]
    ) -> tp.Iterator[mne.io.Raw]:
        for event in events:
            raw = event.read()
            raw = raw.pick(self.pick_types, verbose=False)

            if self.notch_filter is not None:
                raw.load_data()
                raw = self._notch_filter(raw)

            if self.filter is not None:
                raw.load_data()
                raw.filter(
                    self.filter[0], self.filter[1], n_jobs=self.mne_cpus, verbose=False
                )
            if self.apply_hilbert:
                raw.load_data()
                raw = raw.apply_hilbert(envelope=True)

            freq = self._output_frequency(event)  # deals with native/specified frequency
            if freq != event.frequency:
                raw.load_data()
                raw = raw.resample(freq, n_jobs=self.mne_cpus, verbose=False)

            if self.scaler is not None:
                raw.load_data()
                scaler = getattr(sklearn.preprocessing, self.scaler)()
                raw._data = scaler.fit_transform(raw._data.T).T

            if self.apply_proj:
                raw.apply_proj()
            if self.find_bad_channels:
                raw.load_data()
                if self.find_bad_channels == "lof":
                    if self.pick_types == ("meg",):
                        bads = []
                        for ch_ in ["mag", "grad"]:
                            bads += mne.preprocessing.find_bad_channels_lof(
                                raw, picks=ch_
                            )
                    else:
                        bads = mne.preprocessing.find_bad_channels_lof(raw)
                elif self.find_bad_channels == "maxwell":
                    bads, _ = mne.preprocessing.find_bad_channels_maxwell(raw)

                if len(bads) / len(raw.ch_names) > 0.2:
                    logger.warning(
                        f"Found more than 20% bad channels ({len(bads)}/{len(raw.ch_names)}), interpolating them."
                    )
                raw.info["bads"] = list(set(raw.info["bads"] + bads))
                raw.interpolate_bads(reset_bads=True)
            yield raw

    def _get(self, event: ns.events.Meg, start: float, duration: float) -> torch.Tensor:
        start += self.offset

        # Extend window in case of disjoint baseline
        window_start, window_duration = start, duration
        if self.baseline is not None and (
            self.baseline[0] < 0.0 or self.baseline[1] > duration
        ):
            start_offset = min(self.baseline[0], 0.0)
            start += start_offset
            duration = max(duration, self.baseline[1]) - start_offset
            baseline = (max(0.0, self.baseline[0]), self.baseline[1] - start_offset)
        else:
            start_offset = 0.0
            baseline = self.baseline  # type: ignore

        # cached_preprocessing
        # (copy to avoid corrupting cache)
        raw = next(self._get_preprocessed_data([event]))

        assert isinstance(raw, mne.io.BaseRaw)  # for typing
        # safeguard for first_samp
        if raw.first_samp and not event.start:
            raise RuntimeError(
                "event.start should be raw.first_samp / freq for consistency"
            )
        overlap_start, overlap_duration = self._get_overlap(
            event.start, event.duration, start, duration
        )
        data_start = overlap_start - event.start  # time in the M/EEG referential
        freq = self._output_frequency(event)  # deals with native/specified frequency
        # times in time_as_index are assumed to be relative to first_samp (cf doc)
        # time_as_index is slow, so let's do it manually
        # start_idx, stop_idx = raw.time_as_index([meg_start, meg_start + overlap_duration])
        start_idx = freq.to_ind(data_start)
        if start_idx == raw.n_times:
            start_idx -= 1
        # apply freq on overlap to keep always the same size, and minimum to 1
        stop_idx = start_idx + max(1, freq.to_ind(overlap_duration))

        try:
            npdata, _ = raw[:, start_idx:stop_idx]
        except ValueError:
            msg = (
                "Failed to read event %r (start=%s duration=%s)\n"
                "(start_idx=%s stop_idx=%s in %s)"
            )
            logger.warning(msg, event, start, duration, start_idx, stop_idx, raw)
            raise
        data = torch.from_numpy(npdata).float()

        # Apply baseline to the data
        if self.baseline is not None:
            msg = f"unexpected baseline:{baseline}"
            tmin, tmax = [freq.to_ind(t) for t in baseline]
            assert (tmax - tmin) > 0, msg
            data -= data[:, tmin:tmax].mean(1, keepdim=True)

            # Crop larger window in case of disjoint baseline
            if window_start != start or window_duration != duration:
                start, duration = window_start, window_duration
                tmin = freq.to_ind(-start_offset)
                tmax = freq.to_ind(-start_offset + duration)
                data = data[:, tmin:tmax]

                # Recompute start_idx for cropped window
                overlap_start, overlap_duration = self._get_overlap(
                    event.start, event.duration, start, duration
                )
                data_start = overlap_start - event.start
                start_idx = freq.to_ind(data_start)

        # initialize output
        channel_idx = self._get_channels(raw.ch_names)
        n_samples = max(1, freq.to_ind(duration))
        out: torch.Tensor = torch.zeros(
            (len(self._channels), n_samples), dtype=torch.float32
        )
        # get overlap times between output and meg
        out_slice, event_slice = self._get_overlap_slice(
            freq,  # equal to raw.info["sfreq"]
            event.start,
            event.duration,
            start,
            duration,
        )
        event_slice = slice(
            event_slice.start - start_idx,
            event_slice.stop - start_idx,
            event_slice.step,
        )
        if event_slice.stop == data.shape[-1] + 1:
            # rounding for last sample of data, let's repeat the final sample
            es = event_slice
            event_slice = list(range(es.start, es.stop, es.step))  # type: ignore
            if event_slice:  # for empty case
                event_slice[-1] -= 1  # type: ignore
        # set to output
        try:
            out[channel_idx, out_slice] = data[:, event_slice]
        except:
            print(f"Failure with {out_slice=} and {event_slice=}")
            print(f"{start=}, {duration=}, {freq=}")
            print(f"{start_idx=} {stop_idx=}")
            print(f"{data.shape=}\n{event=}\n{raw=}")
            raise

        if self.clamp is not None:
            out = out.clamp(min=-self.clamp, max=self.clamp)

        return out

    def _update_channels(self, ch_names: tp.List[str]) -> None:
        channels = self._channels  # avoid calling pydantic attr too many times
        for ch in ch_names:
            if ch not in self._channels:
                channels[ch] = len(channels)

    def _prepare_channels(self, events: tp.List[ns.events.Meg]) -> None:
        for raw in self._get_preprocessed_data(events):
            self._update_channels(raw.ch_names)

    def _get_channels(self, ch_names: tp.List[str]) -> tp.List[int]:
        if not self._channels:
            self._update_channels(ch_names)
        try:
            channel_idx = [self._channels[ch] for ch in ch_names]
        except KeyError as e:
            raise KeyError(
                f"Channel {e} not found in the channel mapping, likely because this dataset contains recordings with different sets of channel names. Try calling self.prepare on the whole events dataframe."
            ) from e
        return channel_idx

    def _notch_filter(self, raw: mne.io.Raw) -> mne.io.Raw:
        notch_freqs = np.arange(
            self.notch_filter, raw.info["sfreq"] / 2, self.notch_filter
        )  # type: np.ndarray
        logger.info(f"Applying notch filter with notch_freqs={notch_freqs}")
        raw = raw.notch_filter(
            notch_freqs, phase="zero", filter_length="auto", n_jobs=self.mne_cpus
        )
        return raw


class Eeg(Meg):
    name: tp.Literal["Eeg"] = "Eeg"  # type: ignore
    event_types: tp.Literal["Eeg"] = "Eeg"
    pick_types: tuple[str, ...] = pydantic.Field(("eeg",), min_length=1)

    infra: MapInfra = MapInfra(
        timeout_min=120,
        gpus_per_node=0,
        cpus_per_task=10,
        version="1",
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("pybv>=0.7.6",)

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)

        try:
            import pybv  # noqa # pylint: disable=unused-import
        except ModuleNotFoundError as err:
            msg = (
                "Caching EEG timelines in the BrainVision format requires the pybv package."
                "Please install it with `pip install pybv`."
            )
            raise ModuleNotFoundError(msg) from err

    # Redefining _get_preprocessed_data to use the BrainVision format as cache type, which
    # is much faster to load than FIF (e.g. 7x speed up when loading 70k timelines).
    @infra.apply(
        item_uid=lambda e: str(e.filepath),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
        cache_type="MneRawBrainVision",
    )
    def _get_preprocessed_data(self, events: list[ns.events.Meg]) -> tp.Iterator[Raw]:
        return super()._get_preprocessed_data(events)


