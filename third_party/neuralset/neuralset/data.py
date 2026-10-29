# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
import typing as tp
from abc import abstractmethod
from pathlib import Path

import pandas as pd
import pydantic

from .base import PathLike, StrCast, _Module
from .enhancers import Enhancer
from .events import Event
from .infra import CacheDict, MapInfra
from .segments import validate_events
from .utils import compress_string

logger = logging.getLogger(__name__)


def _check_folder_path(path: PathLike, name: str) -> Path:
    """Check that the parent path exists and create directory"""
    path = Path(path)
    if not path.parent.exists():
        raise RuntimeError(f"Parent folder {path.parent} of {name} must exist first.")
    path.mkdir(exist_ok=True)
    return path


def _validate_study_name(name: str) -> None:
    pattern = re.compile(r"^[A-Z][A-Za-z]*?[0-9]{4}(Bold|Beta|Meg|Eeg)?$")
    if pattern.match(name) is None:
        raise ValueError(
            "Study name must CamelCase starting by at least 1 "
            "capitalized letter followed by 4 digits, "
            "(optionally followed by 'Bold', 'Beta', 'Meg' or 'Eeg')\n"
            "Eg: TestMeg2012, Gwilliams2022, Allen2022Bold, etc..."
            f"\nbut got {name!r}\n"
        )


TIMELINES: tp.Dict[str, "BaseData"] = {}
STUDIES: tp.Dict[str, tp.Type["BaseData"]] = {}


def _get_study(name: str) -> tp.Type["BaseData"]:
    """Access the study class un the dict
    If the study is not already present, load all the study files in
    the studies folder and retry
    """
    if name not in STUDIES:
        # load all modules in the studies folder
        for fp in Path(__file__).with_name("studies").glob("*.py"):
            # limit number of loaded files
            if not fp.name.startswith("test_"):
                try:
                    defined = f"class {name}" in fp.read_text()
                except FileNotFoundError:
                    pass  # sometimes new files make a mess with editable_mode=strict install
                else:
                    if defined:
                        importlib.import_module(f"neuralset.studies.{fp.stem}")
    if name not in STUDIES:
        raise ValueError(
            f"Could not find study {name} (currently loaded studies: {list(STUDIES.keys())}).\n"
            "You may need to import the study module beforehand (possibly inline for \n"
            "jobs spawned in another process/cluster to make sure the cache is reloaded \n"
            "within the function)"
        )
    return STUDIES[name]


class BaseData(_Module):
    # Timeline level
    subject: StrCast
    path: PathLike
    timeline: str = ""

    # Study level
    version: tp.ClassVar[str] = "v2"
    study: tp.ClassVar[str]
    url: tp.ClassVar[str] = ""
    bibtex: tp.ClassVar[str] = ""
    licence: tp.ClassVar[str] = ""
    device: tp.ClassVar[str] = ""  # optional if _load_raw not specified
    description: tp.ClassVar[str] = ""

    @classmethod
    @tp.final
    def download(cls, path: PathLike, **kwargs: tp.Any) -> None:
        path = Path(path)
        cls._download(path)
        if not path.exists():
            raise RuntimeError(f"Path does not exist: {path}")
        if not path.is_dir():
            raise RuntimeError(f"Path is not a directory: {path}")
        if not any(path.iterdir()):
            raise RuntimeError(f"Directory is empty: {path}")
        logger.info(f"Success: Study downloaded to {path}.")
        # Set and validate folder permissions
        cmd = f"chmod 777 -R {path}"
        logger.info(f"Setting permissions: {cmd}")
        subprocess.check_output(cmd.split(), shell=False)
        if not oct(os.stat(path).st_mode & 0o777) == "0o777":
            raise RuntimeError(f"Directory permissions not set to 777: {path}")
        logger.info(f"Success: Permissions set to 777 for {path}.")

    @classmethod
    @abstractmethod
    def _download(cls, path: Path) -> None:
        """Download dataset.
        Needs to be overriden by user.
        """
        raise NotImplementedError("Dataset not available to download yet.")

    @classmethod
    @abstractmethod
    def _iter_timelines(cls, path: PathLike) -> tp.Iterator["BaseData"]:
        """Iterate timelines.
        Needs to be overriden by user.
        """
        raise NotImplementedError

    @tp.final  # typing makes sure it's not overriden
    @classmethod
    def iter_timelines(cls, path: PathLike) -> tp.Iterator["BaseData"]:
        path = _check_folder_path(path, name="path")
        study = cls.study
        if path.name.lower() != study.lower():
            # use the subfolder with capitalized or uncapitalized name if it exists,
            # this enables using same folder everywhere
            for name in (study, study.lower()):
                if (path / name).exists():
                    path = path / name
                    logger.debug("Updating study path to %s", path)
                    break
        yield from cls._iter_timelines(path)

    def __init_subclass__(cls) -> None:
        name = cls.__name__
        cls.study = name
        super().__init_subclass__()
        if cls.device not in Event._CLASSES:
            raise RuntimeError(
                f"No device named {cls.device}, available: {list(Event._CLASSES)}"
            )
        if not name.startswith("_"):
            _validate_study_name(name)
            STUDIES[name] = cls

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        # automatic definition of timeline if not specified, as the string
        # concatenation of all init parameters
        if not self.timeline:
            excludes = "path", "timeline"
            timeline = self.study
            for name, arg in self.model_fields.items():
                if name in excludes or arg.init_var is False:
                    continue
                value = getattr(self, name)
                assert value is None or isinstance(value, (str, float, int)), (
                    "Automatic timeline "
                    "assignment is not supported for classes initialized by "
                    f" something else than strings or float but got: "
                    f"{arg}={value} (type: {type(value)}). Specify timeline in "
                    f"the definition of {self.__class__.__name__}."
                )
                timeline += f"_{name}-{str(value)}"
            self.timeline = compress_string(timeline)
        # keep a record of accessible instances
        TIMELINES[self.timeline] = self

    @abstractmethod
    def _load_events(self) -> pd.DataFrame:
        """Needs to be overriden by user."""
        raise NotImplementedError

    @tp.final
    def load(self) -> pd.DataFrame:
        # get study dependent DataFrame
        events = self._load_events()
        # Add timeline information
        for col in ["study", "subject", "timeline"]:
            if col in events:
                raise ValueError(f"Column {col} already exists in the events dataframe")
            events[col] = getattr(self, col)
        # validate time series
        events = validate_events(events)
        return events


class StudyLoader(pydantic.BaseModel):
    """Config for loading a study.
    Once build, just call :code:`cfg.build()` to get the study dataframe.

    Parameters
    ----------
    name: str
        name of the study
    path: Path or str
        path of the study raw data (or folder containing a subfolder named after the
        study)
    query: str or None
        query over the study summary dataframe (see :code:`loader.study_summary()`),
        typically used for debugging to avoid loading all timelines.
        At least one of the following columns must be used in the query: :code:`timeline_index`, :code:`subject_index` and
        :code:`subject_timeline_index` for filtering
        Eg: :code:`"timeline_index < 3"` to query 3 timelines, :code:`"subject=='subject1'"`,
        to query :code:`subject1` only, `:code:`"subject_index < 10"` to query 10 subjects,
        or :code:`"subject_timeline_index < 2"` to query at most 2 timelines per subject.
    enhancers: list of EnhancerConfig
        list of preprocessing steps to apply on the events sequentially
    infra: MapInfra
        infra for the computation, defaulting to using a process pool.
        Activate caching by setting :code:`infra.folder`

    Usage
    ------
    .. code-block:: python

        loader = StudyLoader(
            name=<study name>,
            path=<shared study folder>,
            infra={"folder": <cache folder>}
        )
        events = loader.build()  # will create the events dataframe and cache intermediate data

    Note
    -----
    - all cache will be dumped in a unique specific folder per study
    - :code:`subject` field gets updated to include the study name so as to avoid overlaps
    - setting a deprecated parameter will trigger compatibility mode and use legacy uid even
      though it is not used anymore (eg: max_workers=1 will trigger compatibility mode)

    Deprecations
    ------------
    - :code:`cache` is deprecated and replaced by the :code:`loader.infra.folder`
    - :code:`max_workers` is deprecated in favor of :code:`loader.infra.max_jobs`
    - :code:`download` is deprecated in favor of calling :code:`loader.study().download(folder)`
    - :code:`install` is deprecated in favor of calling :code:`loader.study().install_requirements()`
    - :code:`n_timelines` is deprecated in favor of calling using :code:`loader.query = "index < 12"`
    """

    name: str
    path: PathLike
    query: str | None = None
    # Note: enhancers have a trick to always include discriminator
    enhancers: tp.List[Enhancer] = []
    infra: MapInfra = MapInfra(cluster="processpool", max_jobs=None)
    _build_infra: MapInfra = MapInfra()
    _timelines: tp.List[BaseData] | None = None  # cache
    # deprecated
    n_timelines: int | tp.Literal["all"] = "all"
    cache: PathLike | None = None
    max_workers: int | None = None  # use all cpus by default
    download: bool | None = None
    install: bool | None = None

    def _exclude_from_cls_uid(self) -> tp.List[str]:
        return ["path", "cache", "max_workers", "download", "install", "n_timelines"]

    @pydantic.field_validator("name")
    @staticmethod
    def _is_study_name(name: str) -> str:
        _validate_study_name(name)
        return name

    # pylint: disable=unused-argument
    def model_post_init(self, log__: tp.Any) -> None:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                _ = CacheDict(folder=tmp, cache_type="ParquetPandasDataFrame")
        except ValueError as e:
            raise RuntimeError('Run "pip install pyarrow" to enable study cache') from e
        # handle deprecations !
        if self.download is not None:
            msg = "studyloader.download is deprecated, use studyloader.study.download(studyloader.path)"
            msg += " and do not set 'download' in your config"
            raise ValueError(msg)
        if self.install is not None:
            msg = "studyloader.install is deprecated, use studyloader.study.install_requirements()"
            msg += " and do not set 'install' in your config"
            raise ValueError(msg)
        if self.max_workers is not None:
            msg = "studyloader.max_workers is deprecated, set studyloader.infra.max_jobs instead"
            msg += " and do not set 'max_workers' in your config"
            raise ValueError(msg)
        if self.n_timelines != "all":
            msg = "studyloader.n_timelines is deprecated, only set studyloader.query = 'index < n' instead"
            msg += " and do not set 'n_timelines' in your config"
            raise ValueError(msg)
        if self.cache is not None:
            msg = "studyloader.cache is deprecated, only set studyloader.infra.folder instead"
            msg += " and do not set 'cache' in your config"
            raise ValueError(msg)
        # apply the processing/caching infra to the call method
        study = self.study()  # checking it works
        # set a specific name pattern for cache folder
        name = self.__class__.__name__ + ",{version}"
        i = self.infra  # shortcut
        # deprecate cache if study version is updated:
        i.version = self.model_fields["infra"].default.version + f"-{study.version}"
        folder_name = f"{name},{self.name}"
        i._uid_string = folder_name + "/{method},{uid}"
        # update hidden infra
        names = ["folder", "version", "_uid_string", "mode"]
        self._build_infra._update({x: getattr(i, x) for x in names})
        # if force, clear cache folder manually
        if self.infra.mode == "force" and self.infra.folder is not None:
            folder = Path(self.infra.folder) / folder_name
            if folder.exists():
                shutil.rmtree(folder)

    # API #
    def study(self) -> tp.Type[BaseData]:
        """Returns the study class"""
        study = _get_study(self.name)
        return study

    def iter_timelines(self) -> tp.Iterator[BaseData]:
        """Iterate on the timelines of the study"""
        if self._timelines is None:
            self._timelines = list(self.study().iter_timelines(self.path))
        else:
            for tl in self._timelines:
                TIMELINES[tl.timeline] = tl  # make sure it is registered
        return iter(self._timelines)

    def study_summary(self, apply_query: bool = True) -> pd.DataFrame:
        """Returns a dataframe with 1 row per timeline and study attributes as columns.
        :code:`query` parameter is used on this dataframe for subselection

        Parameter
        ---------
        apply_query: bool
            if False returns the full the summary, otherwise filter it
            according to the query

        Additional field
        ----------------
        :code:`subject_index`: int
            the index of the subject in the study
        :code:`timeline_index`: int
            the index of the timeline in the study (equivalent to "index")
        :code:`subject_timeline_index`: int
            the index of the timeline among a subject's timelines in the study
            (used for querying at most :code:`n` timelines per subjects)
        """
        out = pd.DataFrame([dict(tl) for tl in self.iter_timelines()])
        out["subject"] = out.subject.apply(lambda x: f"{self.name}/{x}")
        if any(n in out.columns for n in ["subject_index", "timeline_index"]):
            msg = "Study dataframes are not allowed to have subject_index nor timeline_index"
            msg += f" in their column, found columns: {list(out.columns)}"
            raise RuntimeError(msg)
        groups = out.groupby("subject")
        out.loc[:, "subject_index"] = groups.ngroup()
        out.loc[:, "subject_timeline_index"] = groups.cumcount()
        out.loc[:, "timeline_index"] = out.index  # type: ignore
        if apply_query and self.query is not None:
            out = out.query(self.query)
        return out

    def build(self) -> pd.DataFrame:
        """Builds the events dataframe after filtering according to the query if provided"""
        # fast registration of all timelines into cache
        # so that they can be used
        for _ in self.iter_timelines():
            pass
        return list(self._build([self.query]))[0]

    @infra.apply(
        item_uid=lambda item: item.timeline,
        exclude_from_cache_uid=("enhancers", "query"),
    )
    def _load_timelines(
        self, timelines: tp.Iterable[BaseData]
    ) -> tp.Iterator[pd.DataFrame]:
        """Loads raw timelines and cache them"""
        for tl in timelines:
            TIMELINES[tl.timeline] = tl  # make sure it is registered
            out = tl.load()
            out.subject = f"{self.name}/{tl.subject}"
            yield out

    @_build_infra.apply(
        item_uid=str,
        exclude_from_cache_uid=("query",),
        # 5x faster write, 3x faster read, 10x smaller compared to CSV:
        cache_type="ParquetPandasDataFrame",
    )
    def _build(self, queries: tp.Iterable[str | None]) -> tp.Iterator[pd.DataFrame]:
        """Loads cached raw timelines, apply enhancers and cache result"""
        timelines = list(self.iter_timelines())
        summary: pd.DataFrame | None = None
        for query in queries:
            sub = timelines
            if query is not None:
                if summary is None:
                    summary = self.study_summary(apply_query=False)
                selected = summary.query(query)
                sub = [timelines[i] for i in selected.index]
            if not sub:
                msg = f"No timeline found for {self.name} with {query=}"
                raise RuntimeError(msg)
            events = pd.concat(list(self._load_timelines(sub))).reset_index(drop=True)
            for enhancer in self.enhancers:
                events = enhancer(events)
            yield events
