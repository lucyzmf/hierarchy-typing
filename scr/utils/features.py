import logging
import typing as tp

import numpy as np
import pandas as pd
import pydantic
import torch
import tqdm
from exca import MapInfra
from scr.utils.feature_dicts import *

import neuralset as ns
from neuralset.features import BaseText

from . import enhance_events  # noqa: F401  (registers the event enhancers named in the configs)


logger = logging.getLogger(__name__)


def clean_accents(text: str) -> str:
    # á, é, í, ó, ú, ü, ñ
    text = text.lower()

    text = text.replace("á", "a")
    text = text.replace("é", "e")
    text = text.replace("í", "i")
    text = text.replace("ó", "o")
    text = text.replace("ú", "u")
    text = text.replace("ü", "u")
    text = text.replace("ñ", "n")

    return text


class Word(ns.events.Word):
    """
    Word event.
    """

    # can be populated by data enhancers
    context: str = ""  # eg causal context
    true_sequence: str = ""  # sentence containing the word
    sentence_char: int | None = None  # character in the sentence
    word_id: int | None = None  # word id in the sentence


class SpacyEmbedding(ns.features.SpacyEmbedding):
    """override spacy embedding to add language"""

    LANGUAGES: tp.ClassVar[tp.Dict[str, str]] = dict(
        english="en_core_web_lg",
        french="fr_core_news_md",
        spanish="es_core_news_md",
        spanish_lg="es_core_news_lg",
        french_lg="fr_core_news_lg",
    )


class Syllable(ns.events.Text):
    """"""

    word: str


class SyllableEmbedding(BaseText):
    """subword spanish embedding using fasttext"""

    name: tp.Literal["SyllableEmbedding"] = "SyllableEmbedding"
    event_types: tp.Literal["Syllable"] = "Syllable"
    language: tp.Literal["spanish", "english"] = "spanish"
    model_name: str = "test"
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("fasttext",)

    @property
    def model(self) -> tp.Any:
        import fasttext

        if not hasattr(self, "_model"):
            self._model = fasttext.load_model(
                f"scr/model_test/{self.model_name}_{self.language}.bin"
            )

        return self._model

    @property
    def subwords(self) -> set:
        unique_words = set(self.model.get_words())
        if not hasattr(self, "_subwords"):
            self._subwords = set()
            for w in unique_words:
                for s in self._model.get_subwords(w)[0]:
                    self._subwords.add(s)
        return self._subwords

    def get_embedding(self, text: str) -> np.ndarray:
        if text in self.subwords:
            subword_id = self.model.get_subword_id(text)
            return self.model.get_input_vector(subword_id)
        else:
            return self.model.get_word_vector(text)


class Button(ns.events.Event):
    button: str = pydantic.Field("", min_length=1)
    is_left_key: bool

    def get_code(self) -> int:
        from scr.utils.feature_dicts import key_to_int

        return key_to_int[self.button]


class ButtonFeat(ns.features.BaseStatic):
    name: tp.Literal["ButtonFeat"] = "ButtonFeat"
    event_types: tp.Literal["Button"] = "Button"
    feature_type: tp.Literal["hand", "class", "is_vowel"] = "hand"

    infra: MapInfra = MapInfra(version="2")

    def _exclude_from_cache_uid(self) -> tp.List[str]:
        return super()._exclude_from_cache_uid() + ["duration", "frequency"]

    def prepare(self, events: pd.DataFrame) -> None:
        # Populate the _get_latents cache for every unique button up-front, so
        # that per-event lookups inside run_subjs subject tasks are pure cache
        # reads rather than racing 1-sample submissions (which caused
        # KeyError: '<button>' on cache misses).
        events_ = self._events_from_dataframe(events)
        self._get_latents(events_)

    def get_embedding(self, event: ns.events.Event) -> np.array:
        if self.feature_type == "class":
            latent = event.get_code()
        elif self.feature_type == "hand":
            latent = int(event.is_left_key)
        elif self.feature_type == "is_vowel":
            latent = int(event.button in "aeiou")
        else:
            raise ValueError("feature type {self.feature_type} is not defined")
        assert type(latent) == int, f"button retrieval from {event} failed"
        return np.array([latent])

    def get_static(self, event: ns.events.Text) -> torch.Tensor:
        latent = torch.from_numpy(next(self._get_latents([event])))
        return latent

    @infra.apply(
        item_uid=lambda event: str(event.button),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
    )
    def _get_latents(self, events: tp.List[ns.events.Text]) -> tp.Iterator[np.ndarray]:
        if len(events) > 1:
            events = tqdm.tqdm(events, desc="Computing word embeddings")  # type: ignore
        for event in events:
            yield self.get_embedding(event)


class ContextWordEmbeddingPinet(ns.features.HuggingFaceText):
    """
    context word embedding class with override of add_context function
    """

    name: tp.Literal["ContextWordEmbeddingPinet"] = "ContextWordEmbeddingPinet"
    contextualized: bool = True


features = tp.Union[
    SyllableEmbedding
    | ButtonFeat
    | SpacyEmbedding
    | ContextWordEmbeddingPinet
    | ns.features.text.WordFrequency
]

Features = tp.Annotated[features, pydantic.Field(discriminator="name")]
