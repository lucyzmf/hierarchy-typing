# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Event handling classes and functions.

The class `Event` and its children (e.g. `Sound`, `Word`, etc.) define the
expected fields for each event type.
"""

import functools
import logging
import typing as tp
import urllib
from abc import abstractmethod
from pathlib import Path

import pandas as pd
import pydantic

from .base import Frequency, StrCast, _Module
from .utils import ignore_all, warn_once

E = tp.TypeVar("E", bound="Event")
logger = logging.getLogger(__name__)


class Event(_Module):
    """Base class for all event types with the bare minimum common fields.

    If the event is instantiated with `from_dict()`, additional non-required
    fields that are provided will be ignored instead of causing an error.
    """

    start: float
    timeline: str
    duration: pydantic.NonNegativeFloat = 0.0
    extra: dict[str, tp.Any] = {}
    type: tp.ClassVar[str] = "Event"
    _CLASSES: tp.ClassVar[dict[str, tp.Type["Event"]]] = {}

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        # register all events
        cls.type = cls.__name__
        Event._CLASSES[cls.__name__] = cls

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if pd.isna(self.start):
            raise ValueError(f"Start time needs to be provided for {self!r}")

    @classmethod
    def from_dict(cls: tp.Type[E], row: tp.Any) -> E:
        """Create event from dictionary/row/named-tuple while ignoring non-specified fields.
        Keys starting with "extra__" (double _) are packed into the "extra" dictionary field.
        """
        if hasattr(row, "_asdict"):
            row = row._asdict()  # supports named tuples
        cls_ = cls._CLASSES[row["type"]]
        if not issubclass(cls_, cls):
            raise TypeError(f"{cls_} is not a subclass of {cls}")
        fs = set(cls_.model_fields)  # type: ignore
        kwargs: dict[str, tp.Any] = {}
        extra = {}
        for k, v in row.items():
            if k in fs:
                kwargs[k] = v
            elif k.startswith("extra__") and not pd.isna(v):
                extra[k[7:]] = v
        kwargs.setdefault("extra", {}).update(extra)
        for key, val in kwargs.items():
            # dataframe can replace empty strings or 0 values by nan
            # this is particularly the case for duration field, and language for Word
            if pd.isna(val):
                field = cls_.model_fields[key]  # type: ignore
                default = field.default
                null_default = not field.is_required() and (
                    not default or default is None
                )
                cases = [
                    x
                    for cls in (str, float, int, pydantic.NonNegativeFloat)
                    for x in (cls, cls | None)
                ]
                if null_default and field.annotation in cases:
                    kwargs[key] = field.default
        try:
            out = cls_(**kwargs)
        except Exception as e:
            logger.warning(
                "Event.from_dict parsing failed for input %s\nmapped to %s\n with error: %s)",
                row.to_string() if hasattr(row, "to_string") else row,
                kwargs,
                e,
            )
            raise
        return out

    def to_dict(self) -> dict[str, tp.Any]:
        """Export the event as a dictionary usable for csv dump.
        "extra" dictionary field is flatened into "extra.subfield" fields to simplify
        queries through pandas.
        """
        # avoid Path in exports
        out = {x: str(y) if isinstance(y, Path) else y for x, y in self if x != "extra"}
        out["type"] = self.type
        out.update({"extra__" + x: y for x, y in self.extra.items()})
        return out

    @property
    def stop(self) -> float:
        return self.start + self.duration


Event._CLASSES["Event"] = Event


class BaseDataEvent(Event):
    """A base class for events who's data needs to be read from a file."""

    filepath: Path | str = ""
    frequency: float = 0
    _read_method: tp.Any = None

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if not self.filepath:
            raise ValueError("A filepath must be provided")
        # check whether file actually points to register
        self._set_read_method()
        fp = str(self.filepath)
        self.filepath = fp
        if ":" not in str(fp):  # deactivate check
            # make sure to store file as a string in dataframe
            if not Path(fp).exists():
                warn_once(f"file missing: {fp}")

    def _set_read_method(self) -> None:
        try:
            if getattr(self, "_read_method", None) is not None:
                return
        except TypeError:  # pydantic bugs with private attr before model_post_init
            pass  # https://github.com/pydantic/pydantic/issues/9098
        tag = "method:"
        fp = str(self.filepath)
        if not fp.startswith(tag):
            self._read_method = self._read
            return
        # Store read method for reuse in subprocesses (where TIMELINES may not be filled)
        # avoid circular import:
        from .data import TIMELINES  # pylint: disable=import-outside-toplevel

        components = urllib.parse.urlparse(fp)
        assert components.netloc == ""
        assert components.params == ""
        assert components.fragment == ""
        # use a specific loader
        inst = TIMELINES[self.timeline]
        kwargs = dict(urllib.parse.parse_qsl(components.query, strict_parsing=True))
        self._read_method = functools.partial(getattr(inst, components.path), **kwargs)

    def __hash__(self) -> int:
        """required for lru_cache"""
        return hash(self.to_dict())

    def __eq__(self, other: tp.Any) -> bool:
        """required for lru_cache"""
        if isinstance(other, self.__class__):
            return self.__hash__() == other.__hash__()
        return False

    def read(self) -> tp.Any:
        self._set_read_method()
        return self._read_method()

    @abstractmethod
    def _read(self) -> tp.Any:
        return

    def _missing_duration_or_frequency(self) -> bool:
        return any(not x or pd.isna(x) for x in [self.duration, self.frequency])


class BaseText(Event):
    """
    Base class for text events.
    """

    language: str = ""
    text: str = pydantic.Field("", min_length=1)


class Text(BaseText):
    """Possibly multi-sentence text"""

    language: str = ""
    text: str = pydantic.Field(..., min_length=1)


class Sentence(BaseText):
    """
    Sentence event.
    """


class Word(BaseText):
    """
    Word event.
    """

    # can be populated by data enhancers
    context: str = ""  # eg causal context
    sentence: str = ""  # sentence containing the word
    sentence_char: int | None = None  # character in the sentence


class Button(Text):
    """"""


class Meg(BaseDataEvent):
    """Brain Meg event"""

    subject: StrCast = ""

    def model_post_init(self, log__: tp.Any) -> None:
        self.subject = self.subject
        if self._missing_duration_or_frequency():
            raw = self.read()
            self.duration = raw.times[-1] - raw.times[0]
            self.frequency = Frequency(raw.info["sfreq"])
            if raw.first_samp > 0 and not self.start:
                start = raw.first_samp / self.frequency
                msg = f"Meg event start for timeline {self.timeline} is 0 while "
                msg += f"raw.first_samp = {raw.first_samp} > 0\n"
                msg += f"(start should have been defined as raw.first_samp / raw.info['sfreq'] = {start})"
                raise ValueError(msg)
        if not self.subject:
            raise ValueError("Missing 'subject' field")
        super().model_post_init(log__)

    def _read(self) -> tp.Any:
        import mne

        with ignore_all():
            return mne.io.read_raw(self.filepath)


class Eeg(Meg):
    """Brain Eeg event"""


