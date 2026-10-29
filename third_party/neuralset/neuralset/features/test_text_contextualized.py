"""Regression test for the contextualized target-word slicing of HuggingFaceText.

The target word's hidden states must be the last *n_target_tokens* of the tokenized
context. An earlier version used the word's *character* count as the number of tokens,
which, for any word longer than its token count, selected (part of) the preceding
context instead.
"""

import numpy as np
import pytest
import torch
from transformers import AutoModel, AutoTokenizer

import neuralset as ns

CONTEXT = "we urgently need internationalization"


@pytest.mark.parametrize("word, position", [("internationalization", "first"), ("internationalization", "last")])
def test_contextualized_slice_is_the_target_word(word: str, position: str) -> None:
    feature = ns.features.HuggingFaceText(
        model_name="gpt2", device="cpu", contextualized=True, cache_all_layers=True, token_aggregation=position
    )
    event = ns.events.Word(text=word, start=0, duration=1, timeline="x", context=CONTEXT)
    out = np.asarray(next(iter(feature._get_latents([event]))))  # no cache folder: computed directly

    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModel.from_pretrained("gpt2").eval()
    ids = tok(CONTEXT, add_special_tokens=False, return_tensors="pt")
    n_prefix = len(tok.encode(CONTEXT[: -len(word)].rstrip(), add_special_tokens=False))
    n_target = ids["input_ids"].shape[1] - n_prefix
    assert n_target == 2 and n_target < len(word)  # the case the old bug got wrong
    with torch.no_grad():
        states = torch.stack(model(**ids, output_hidden_states=True).hidden_states)[:, 0]  # layers x tokens x dim
    expected = states[:, n_prefix] if position == "first" else states[:, -1]
    np.testing.assert_allclose(out, expected.numpy(), rtol=1e-4, atol=1e-4)
