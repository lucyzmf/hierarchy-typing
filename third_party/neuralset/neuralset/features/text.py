# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import typing as tp
from abc import abstractmethod

import numpy as np
import pandas as pd
import pydantic
import torch
import tqdm
from exca import MapInfra
from exca.utils import environment_variables
from torch import nn
from torch.utils.data import DataLoader, Dataset

import neuralset as ns
from neuralset import utils
from neuralset.features.base import BaseStatic

# pylint: disable=attribute-defined-outside-init
# pylint: disable=unused-variable


class BaseText(BaseStatic):
    """
    Base class for text features.
    """

    language: str = "english"
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("Levenshtein",)

    # for precomputing/caching
    infra: MapInfra = MapInfra(version="1")

    def _exclude_from_cache_uid(self) -> tp.List[str]:
        return super()._exclude_from_cache_uid() + ["duration", "frequency"]

    def prepare(self, events: pd.DataFrame) -> None:
        events_ = self._events_from_dataframe(events)
        self._get_latents(events_)

    @infra.apply(
        item_uid=lambda event: str(event.text),
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
        cache_type="MemmapArrayFile",
    )
    def _get_latents(self, events: tp.List[ns.events.Text]) -> tp.Iterator[np.ndarray]:
        if len(events) > 1:
            events = tqdm.tqdm(events, desc="Computing word embeddings")  # type: ignore
        for event in events:
            yield self.get_embedding(event.text)

    def get_static(self, event: ns.events.Text) -> torch.Tensor:
        latent = torch.from_numpy(next(self._get_latents([event])))
        return latent

    @abstractmethod
    def get_embedding(self, text: str) -> np.ndarray:
        raise NotImplementedError


class WordFrequency(BaseText):
    """
    Get word frequency from wordfreq package.
    """

    name: tp.Literal["WordFrequency"] = "WordFrequency"
    event_types: tp.Literal["Word"] = "Word"
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("wordfreq",)
    LANGUAGES: tp.ClassVar[tp.Dict[str, str]] = dict(
        english="en", french="fr", spanish="es", dutch="nl"
    )

    def get_embedding(self, text: str) -> np.ndarray:
        from wordfreq import zipf_frequency  # noqa

        value = zipf_frequency(text, self.LANGUAGES.get(self.language, self.language))
        return np.array([value])


class SpacyEmbedding(BaseText):
    """
    Get word embedding from spacy.
    """

    name: tp.Literal["SpacyEmbedding"] = "SpacyEmbedding"
    event_types: tp.Literal["Word"] = "Word"
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("spacy>=3.8.2",)
    LANGUAGES: tp.ClassVar[tp.Dict[str, str]] = dict(
        english="en_core_web_lg", french="fr_core_news_lg", spanish="es_core_news_lg"
    )

    def get_embedding(self, text: str) -> np.ndarray:
        model = utils.get_spacy_model(model=self.LANGUAGES[self.language])  # lru cached
        vector = model(text).vector
        return vector


class TextDataset(Dataset):
    """
    Dataset for contextual embeddings.
    """

    def __init__(self, events: tp.List[ns.events.Word]):
        self.events = events

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        sel = self.events[idx]
        return sel.text, sel.context


class HuggingFaceText(BaseStatic):
    """
    Get embeddings from HuggingFace language models.
    This feature can be applied to any kind of event which has a text attribute: Word, Sentence, etc.

    Parameters
    ----------
    model_name: str
        Name of the model to use.
    device: str
        Device to use for the model:
        - cpu: for cpu computation
        - cuda: for using gpu0
        - auto: to use gpu if available else cpu
        - accelerate: to use huggingface accelerate (maps to multiple-gpus + use float16)
    layers: float | List[float]
        Layer(s) to use for the embeddings. These are expressed as floats between 0 and 1, where 0 is the first layer and 1 is the last layer.
    token_aggregation: str
        How to aggregate the tokens together. Can be "mean", "sum", "first", "last".
    batch_size: int
        Batch size for the language model.
    contextualized: bool
        If True, the context of the event is used to compute the embeddings.

    Note
    ----
    The tokenizer truncates the input to the maximum size specified by the model
    """

    name: tp.Literal["HuggingFaceText"] = "HuggingFaceText"

    # class attributes
    event_types: tp.Literal["Word", "Sentence"] = "Word"
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("transformers>=4.29.2",)
    infra: MapInfra = MapInfra(
        timeout_min=25,
        gpus_per_node=1,
        cpus_per_task=10,
        min_samples_per_job=4096,
        version="v4",
    )

    # feature attributes
    model_name: str = "gpt2"
    device: tp.Literal["auto", "cpu", "cuda", "accelerate"] = "auto"
    layers: float | tp.List[float] = 2 / 3
    cache_all_layers: bool = False
    token_aggregation: tp.Literal["mean", "sum", "first", "last"] = "mean"
    batch_size: int = 32
    contextualized: bool = False
    pretrained: bool = True

    # initialized later
    _model: nn.Module = pydantic.PrivateAttr()
    _tokenizer: nn.Module = pydantic.PrivateAttr()
    _pad_id: int = pydantic.PrivateAttr()

    @classmethod
    def _exclude_from_cls_uid(cls) -> tp.List[str]:
        return super()._exclude_from_cls_uid() + [
            "device",
            "batch_size",
            "cache_all_layers",
        ]

    def _exclude_from_cache_uid(self) -> tp.List[str]:
        prev = super()._exclude_from_cache_uid()
        if self.cache_all_layers:
            prev += ["layers"]
        return prev + ["frequency", "duration", "device", "batch_size"]

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def prepare(self, events: pd.DataFrame) -> None:
        events_ = self._events_from_dataframe(events)
        self._get_latents(events_)

    @property
    def model(self) -> nn.Module:
        if not hasattr(self, "_model"):
            from transformers import AutoModel, AutoTokenizer

            kwargs: tp.Dict[str, tp.Any] = {}
            if self.model_name.lower().startswith("microsoft/phi"):
                kwargs["trust_remote_code"] = True
            # make sure to truncate on the left side!
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, truncation_side="left", **kwargs
            )
            if "t5" in self.model_name or "bert" in self.model_name:
                from transformers import AutoModelForTextEncoding

                Model = AutoModelForTextEncoding
            elif "Phi-3" in self.model_name:
                from transformers import AutoModelForCausalLM

                Model = AutoModelForCausalLM
            elif "Llama-3.2-11B-Vision" in self.model_name:
                from transformers import MllamaForConditionalGeneration

                Model = MllamaForConditionalGeneration
            else:
                Model = AutoModel
            # instantiate
            if self.device == "accelerate":
                kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
            self._model = Model.from_pretrained(self.model_name, **kwargs)
            if not self.pretrained:
                # device_map = {name: param.device for name, param in self._model.named_parameters()}
                self._model = Model.from_config(self._model.config)
                if self.device == "accelerate":
                    # dump and reload to use device_map="auto"
                    with tempfile.TemporaryDirectory() as tmp:
                        self._model.save_pretrained(tmp)
                        self._model = Model.from_pretrained(tmp, **kwargs)
            if self.device != "accelerate":
                self._model.to(self.device)
            self._model.eval()
            # tokens
            if self._tokenizer.pad_token is None:
                # previously:
                # self._tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                # self._model.resize_token_embeddings(len(self._tokenizer))
                # simpler to use existing EOS token:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._pad_id = self.tokenizer.eos_token_id
        return self._model

    @property
    def tokenizer(self) -> nn.Module:
        self.model
        return self._tokenizer

    def aggregate_layers(self, latents: torch.Tensor) -> torch.Tensor:
        layers = self.layers if isinstance(self.layers, list) else [self.layers]
        n_layers = latents.shape[0]
        assert all(0 <= l <= 1 for l in layers), "Layers must be between 0 and 1"
        layer_indices = np.unique(
            [int(i * n_layers - 1e-6) for i in layers]
        ).tolist()  # 1e-6 to avoid taking index n_layers

        return latents[layer_indices].mean(dim=0)  # type: ignore

    def aggregate_tokens(self, latents: torch.Tensor) -> torch.Tensor:
        if self.token_aggregation == "first":
            out = latents[:, 0, :]  # get CLS token for models like BERT
        elif self.token_aggregation == "last":
            out = latents[:, -1, :]
        elif self.token_aggregation == "mean":
            out = latents.mean(dim=1)
        elif self.token_aggregation == "sum":
            out = latents.sum(dim=1)
        else:
            raise ValueError(f"Unknown token aggregation: {self.token_aggregation}")
        return out

    def get_static(self, event: ns.events.Word) -> torch.Tensor:
        latent = torch.Tensor(next(self._get_latents([event])))
        if self.cache_all_layers:
            return self.aggregate_layers(latent)
        else:
            return latent

    @infra.apply(
        item_uid=lambda event: f"{event.text}_{event.context}",
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
        cache_type="MemmapArrayFile",
    )
    def _get_latents(self, events: tp.List[ns.events.Word]) -> tp.Iterator[np.ndarray]:
        dataset = TextDataset(events)
        dloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Processing the data in batches
        if len(dloader) > 1:
            dloader = tqdm.tqdm(dloader, desc="Computing word embeddings")  # type: ignore
        device = "auto" if self.device == "accelerate" else self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        for target_words, context in dloader:
            # tokenize context
            with environment_variables(TOKENIZERS_PARALLELISM="false"):
                text = context if self.contextualized else target_words
                if isinstance(text, tuple):
                    # temporary fix for tokenizers==0.20.2
                    # https://github.com/huggingface/tokenizers/issues/1672
                    text = list(text)
                if not all(text):
                    msg = f"Empty text or context for target_words {target_words!r}"
                    raise ValueError(msg)
                inputs = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,  # beware to have set truncation_side="left" in init
                ).to(device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            if "hidden_states" in outputs:
                states = outputs.hidden_states
            else:  # bart (encoder/decoder)
                states = outputs.encoder_hidden_states + outputs.decoder_hidden_states
            hidden_states = torch.stack([layer.cpu() for layer in states])
            n_layers, n_batch, n_tokens, n_dims = hidden_states.shape  # noqa

            # -- for each target word, remove padding, and sum hidden states
            for i, target_word in enumerate(target_words):
                # select batch element
                hidden_state = hidden_states[:, i]  # n_layers x tokens x embd

                # count number of pads
                n_pads = sum(inputs["input_ids"][i].cpu().numpy() == self._pad_id)

                if n_pads:
                    hidden_state = hidden_state[:, :-n_pads]

                # select tokens that belong to the target word
                if self.contextualized:
                    # inputs already has the tokenized context; subtract
                    # the prefix token count to get the target word's tokens
                    prefix = context[i][: -len(target_word)].rstrip()
                    n_prefix = (
                        len(self.tokenizer.encode(prefix, add_special_tokens=False))  # type: ignore[operator]
                        if prefix
                        else 0
                    )
                    n_target = n_tokens - n_pads - n_prefix
                    word_state = hidden_state[:, -max(1, n_target) :]
                else:
                    word_state = hidden_state
                word_state = self.aggregate_tokens(word_state)  # layers x embd

                if self.cache_all_layers:
                    yield word_state.cpu().float().numpy()
                else:
                    yield self.aggregate_layers(word_state).cpu().float().numpy()
