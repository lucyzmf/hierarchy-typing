# neuralset (vendored snapshot for the PiNet typing study)

This directory is a trimmed copy of [neuralset](https://github.com/facebookresearch/neuroai)
(Meta FAIR) as of February 2025, the version this study was run with. It is vendored rather
than installed from PyPI because the public releases (0.1+) have a different API and cache
format; the code that runs here is the code that produced the published results.

Only the code this study uses is kept (unused modules, classes and helpers were removed).

Changes relative to the February 2025 upstream snapshot:
- `features/text.py`, `HuggingFaceText` with `contextualized=True`: the target word's hidden
  states are selected by counting the context prefix's tokens (previously the word's
  *character* count was used as a token count). Cache version bumped to `v4`; states are cast
  to float32 before converting to numpy. Upstream fixed the same bug in later releases.
- `features/neuro.py`: `Meg.find_bad_channels` (`"lof"` or `"maxwell"`) option.
- `studies/pinet2024.py`: the study definition for this dataset.

Caching and job infrastructure: [exca](https://facebookresearch.github.io/exca/).

License: MIT (see `LICENSE` in this directory), as for the public neuralset releases.
