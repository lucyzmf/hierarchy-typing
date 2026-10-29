# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import logging
import typing as tp

import numpy as np
import pandas as pd

from .events import Event
from .utils import warn_once

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Segment:
    """What gets out of a `ns.segments.list_segments(events, duration=1.)`.
    Fields:
    - df: pd.DataFrame (dataframe of events occuring in this time segment)
    - start: float (start time of the time segment)
    - duration: float (duration of the time segment)

    Additionally there is a lazily computed and cached "events" field
    as a list of Event instances, for faster feature processing
    """

    # adding a dict interface to this class confuses pytorch-lightning
    # so better avoid it and keep to a standard dataclass

    events: pd.DataFrame
    start: float
    duration: float
    _trigger: float | tp.Dict[str, tp.Any] | None = None
    _event_list: tp.List["Event"] | None = dataclasses.field(
        init=False, repr=False, default=None
    )

    @property
    def event_list(self) -> tp.List["Event"]:
        """cached list of event instances for faster processing"""
        if self._event_list is None:
            # itertuple is so slow for small lengths
            e = self.events
            n = len(e)
            iterable = (e.iloc[k, :] for k in range(n)) if n <= 2 else e.itertuples()
            self._event_list = [
                Event.from_dict(r) for r in iterable if r.type in Event._CLASSES  # type: ignore
            ]
        return self._event_list

    @property
    def stop(self) -> float:
        return self.start + self.duration

    def asdict(self) -> tp.Dict[str, tp.Any]:
        """Allows for feature(**segment.asdict())"""
        return {"events": self.events, "start": self.start, "duration": self.duration}


def _validate_event(event: pd.Series) -> dict[str, tp.Any]:
    """Validate event, i.e. check fields and values are as expected,
    and update it accordingly.

    This is done by instantiating an event object of the corresponding
    type, which carries out the validation, and then updating the input
    with the applied changes (if any).
    """
    # Check types are valid
    event_type = event["type"]
    lower = {x.lower() for x in Event._CLASSES}
    if event_type in Event._CLASSES:
        event_class = Event._CLASSES[event_type]
        event_obj = event_class.from_dict(event).to_dict()

        # Add back fields that were ignored by the Event class
        # segment.update(asdict(event_obj))
        # Very slow, use dict updating instead
        event_dict = {**event, **event_obj}
    elif event_type in lower:
        raise ValueError(f"Legacy uncapitalized event {event}")
    else:
        warn_once(
            f'Unexpected type "{event["type"]}". Support for new event '
            "types can be added by creating new `Event` classes in "
            "`neuralset.events`."
        )
        event_dict = {**event}

    return event_dict


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    """Validate the DataFrame of events (not inplace).

    Returns
    -------
    pd.DataFrame
        DataFrame in which each row has been validated and updated.
    """
    if not events.empty:
        msg = 'events DataFrame must have a "type" column with strings'
        assert "type" in events.keys(), msg
        types = events["type"].unique()
        assert all(isinstance(typ, str) for typ in types), msg
        # event-level validation
        df = pd.DataFrame(
            events.apply(_validate_event, axis=1).tolist(),
            index=events.index,
        )
        # add dynamic field
        df = df.assign(stop=lambda x: x.start + x.duration)

        return df
    else:
        return events.copy()


def intersection_segments(
    events: pd.DataFrame,
    starts: float | np.ndarray,
    durations: float | np.ndarray,
    within_only: bool = False,
    strict_overlap: bool = True,
) -> tp.Generator[Segment, None, None]:
    """Given a start (or array of starts) and duration (or array of durations) providing
    one or more time windows, yield one Segment comprising all events occuring (strictly or not)
    within each time window.
    """
    if events.timeline.nunique() != 1:
        raise RuntimeError("only support a single timeline")
    starts = np.ravel(starts)
    if isinstance(durations, (list, tuple)):
        durations = np.array(durations)
    if not isinstance(durations, np.ndarray):
        durations = durations * np.ones_like(starts)
    stops = np.array(starts + durations)
    starts = starts[:, None]
    stops = stops[:, None]
    estarts = np.array(events.start)[None, :]
    estops = np.array(events.start + events.duration)[None, :]
    # compute selected items for each time window!
    if within_only:
        select = estarts >= starts
        select &= estops <= stops
    else:
        if strict_overlap:
            select = estarts < stops
            select &= estops > starts
        else:
            select = estarts <= stops
            select &= estops >= starts
    for k, (start_, duration) in enumerate(zip(starts, durations)):
        start = float(start_.item())
        yield Segment(events=events.loc[select[k]], start=start, duration=duration)


def _prepare_strided_windows(
    start: float,
    stop: float,
    stride: float,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-8
    starts = np.arange(
        start, stop - duration + eps, stride
    )  # Ignore windows that don't completely overlap like in `mne.events_from_annotations`
    # with `chunk_duration` other than None.
    durations = np.ones_like(starts) * duration
    return starts, durations


def _iter_segments(
    df: pd.DataFrame,
    idx: int | pd.Series | None = None,
    *,
    start: float = 0.0,
    duration: float | tuple[float, float] | None = None,
    stride: float | None = None,
    within_only: bool = False,
    strict_overlap: bool = True,
) -> tp.Generator[Segment, None, None]:
    """
    Make an iterator of segments based on specific events (`idx`), a `stride`, or both.

    See `ns.segments.list_segments` for a description of parameters.
    """
    # make sure the selected events match the dataframe
    events = {}
    # create events_list once and for all
    for row in df.itertuples(index=True):
        if row.type in Event._CLASSES:
            events[row.Index] = Event.from_dict(row)

    # Regular division of timeline
    if stride is not None:
        assert isinstance(stride, (int, float))
        assert isinstance(duration, (int, float))
        stride = float(stride)
        duration = float(duration)

    # Specific events
    if idx is not None:

        # ensure index is a pd.Series
        if isinstance(idx, int):
            idx = df.index == idx  # type: ignore

        if not np.any(idx):
            avail = pd.unique(df["type"])
            raise ValueError(
                "Empty trigger events provided to list_segments (first argument)\n"
                f"Available events.type: {avail} (did you forget capitalizing the event name?)"
            )

        # convert value-based index to boolean-based index
        # caution: "idx.dtype is bool" doesnt work anymore when reloaded
        # from parquet, which gets a weird type-like object as type
        if "bool" in str(idx.dtype).lower():  # type: ignore
            idx = df.loc[idx].index  # type: ignore
        # check index
        df.loc[idx]  # pylint: disable=pointless-statement

        assert isinstance(start, (int, float))
        start = float(start)

    triggers: tp.Generator | list | np.ndarray
    for tl_name, tl in df.groupby("timeline", sort=False):
        # If we select the batch based on existing events
        if idx is not None:
            j = tl.index.isin(idx)
            if not any(
                j
            ):  # Can happen if timeline does not contain any event of interest
                logger.warning(f"No valid events found for timeline {tl_name}.")
                continue

            if stride is None:
                starts: tp.Any = tl.loc[j].start + start
                # cant pickle named tuples, and iterrows is too slow:
                triggers = (r._asdict() for r in tl.loc[j].itertuples())  # type: ignore

                # If duration is not specified, use the duration of the selected events
                if duration is None:
                    durations: tp.Any = tl.loc[j].duration
                elif isinstance(duration, tuple):
                    # sample between min and max
                    durations = np.random.uniform(duration[0], duration[1], len(starts))
                else:
                    durations = np.ones_like(starts) * duration

            else:  # Extract sliding windows within each event
                assert start == 0.0
                starts, durations, triggers = [], [], []
                for row in tl.loc[j].itertuples():
                    assert isinstance(row.start, float) and isinstance(row.stop, float)
                    assert isinstance(duration, float)
                    _starts, _durations = _prepare_strided_windows(
                        row.start,
                        row.stop,
                        stride,
                        duration,
                    )
                    starts.append(_starts)
                    durations.append(_durations)
                    triggers.extend([row._asdict()] * len(_starts))  # type: ignore
                starts = np.concatenate(starts)
                durations = np.concatenate(durations)

        # if we select the batch based on a sliding window
        elif stride is not None:
            assert start == 0.0
            assert isinstance(duration, float)
            starts, durations = _prepare_strided_windows(
                tl.start.min(), tl.stop.max(), stride, duration
            )
            triggers = starts

        else:
            # find from a specific time
            assert duration is not None
            assert df.timeline.nunique() == 1
            starts = [start]
            durations = [duration]
            triggers = starts

        # For each segment
        inter_segments = intersection_segments(
            tl,
            starts,
            durations,
            within_only=within_only,
            strict_overlap=strict_overlap,
        )
        # add triggers and events
        for segment, trigger in zip(inter_segments, triggers, strict=True):
            segment._trigger = trigger
            # find corresponding events and preadd them (muuuch faster than online)
            evlist = [events[i] for i in segment.events.index if i in events]
            # quickcheck (make sure there's not an indexing issue)
            if evlist:
                estop = evlist[0].start + evlist[0].duration
                if evlist[0].start > segment.stop or estop < segment.start:
                    raise RuntimeError("Event list attribution to segment failed")
            segment._event_list = evlist
            yield segment


def iter_segments(
    events: pd.DataFrame,
    idx: pd.Series | None = None,
    *,
    start: float = 0.0,
    duration: float | tuple[float, float] | None = None,
    stride: float | None = None,
    strict_overlap: bool = True,
) -> tp.Generator[Segment, None, None]:
    """
    Yield segments based on events and/or a stride.

    See `ns.segments.list_segments` for description of parameters.
    """
    for segment in _iter_segments(
        events,
        idx=idx,
        start=start,
        duration=duration,
        stride=stride,
        within_only=False,
        strict_overlap=strict_overlap,
    ):
        yield segment


def list_segments(
    events: pd.DataFrame,
    idx: pd.Series | None = None,
    *,
    start: float = 0.0,
    duration: float | tuple[float, float] | None = None,
    stride: float | None = None,
    strict_overlap: bool = True,
) -> list[Segment]:
    """
    Make a list of segments:
    - based on specific events (a single segment is extracted by event):
        ns.segments.list_segments(df, idx=df.type == "Image")
    - based on sliding windows (entire timeline will be subdivided into potentially
        overlapping segments):
        ns.segments.list_segments(df, stride=1.5, duration=3.)
    - or based on both a list of segments and sliding windows (each event will be subdivided
        into potentially overlapping segments; a window must be fully overlapping with the event
        to be valid):
        ns.segments.list_segments(df, idx=df.type == "Image", stride=1.5, duration=3.)

    Parameters
    ----------
    idx: pd.Series
        If provided, list of events to use for defining the segments.
    start: float
        Start time (in seconds) of the segment, with respect to the reference event (or stride).
        E.g. use -1.0 if you want the segment to start 1s before the event.
    duration: optional float
        Duration (in seconds) of the segment (defaults to event duration if only using `idx` to
        extract segments based on specific events).
    stride: optional float
        Stride (in seconds) to use to define sliding window segments.
    """
    return list(
        iter_segments(
            events,
            idx=idx,
            start=start,
            duration=duration,
            stride=stride,
            strict_overlap=strict_overlap,
        )
    )


def find_incomplete_segments(
    segments: tp.Sequence[Segment], event_types: tp.Sequence[tp.Type[Event]]
) -> tp.List[int]:
    """
    Return the indices of segments that do not contain one of the specified event types.
    """
    all_invalid_indices = set()
    for event_type in event_types:
        invalid_indices = set()
        subclasses = [
            name for name, cls in Event._CLASSES.items() if issubclass(cls, event_type)
        ]
        for i, segment in enumerate(segments):
            if not any(segment.events.type.isin(subclasses)):
                invalid_indices.add(i)
        if invalid_indices:
            msg = f"{len(invalid_indices)} segments out of {len(segments)} did not contain valid events for event type {event_type}"
            logger.warning(msg)
        all_invalid_indices.update(invalid_indices)
    return sorted(list(all_invalid_indices))


