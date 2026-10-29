# A hierarchy of dynamic neural codes for typing

Code for

> *Zhang, M., Lévy, J., d'Ascoli, S., Rapin, J., Alario, F., Bourdillon, P., ... & King, J. R. (2025). From thought to action: 
> How a hierarchy of neural dynamics supports language production. arXiv preprint arXiv:2502.07429.*

We decode four levels of language representation (context, word, syllable, letter) from
MEG and EEG recorded while 35 skilled typists type memorised Spanish sentences, and show that
they are produced top-down, overlap in time, and are held by dynamic neural codes whose speed
depends on their level in the hierarchy.

## Contents

```
scr/
  main_hierar.py        grid: hierarchy decoding (paper Fig 2A-B, S3)
  main_fig2_grid.py     grid: successive keys (Fig 2D, S4)
  main_seq.py           grid: temporal generalisation / dynamic codes (Fig 3, 4, S5, S6)
  fig*.py               figures, drawn from the cached grid results
  configs/              experiment configurations (defaults.py: paths & compute, from env vars)
  experiments/          decoding experiments (cached per subject with exca)
  utils/                features, decoders, metrics, event enhancers
third_party/neuralset/  vendored February-2025 neuralset (see third_party/neuralset/README.md)
scripts/                build_events.sh, run_grids.sh, make_figures.sh, download_fasttext.sh
assets/                 syllable segmentation of the stimulus words, accent-stripped word list
```

## Install

Requires Python 3.11-3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync      # exact environment from uv.lock, including the spaCy model es_core_news_lg 3.8.0
```

GPT-2 (context embeddings) is downloaded from Hugging Face on first use. The Spanish
fastText model used for syllable embeddings (7.2 GB,
[Lucyzmf/pinet-fasttext-es](https://huggingface.co/Lucyzmf/pinet-fasttext-es)) is fetched
and checksummed with:

```bash
uv run bash scripts/download_fasttext.sh   # -> scr/model_test/wiki_all_minn1_spanish.bin
```

How it was trained: see the [model card](https://huggingface.co/Lucyzmf/pinet-fasttext-es).

## Data

MEG and EEG recordings with behavioural logs:
[huggingface.co/datasets/bcbl190626/SpanishBCBL](https://huggingface.co/datasets/bcbl190626/SpanishBCBL)
(CC BY-NC 4.0, ~262 GB).

```bash
uv run hf download bcbl190626/SpanishBCBL --repo-type dataset --local-dir /path/to/SpanishBCBL
```

The dataset root must contain `MEG/FIF`, `MEG/logs`, `EEG/EEG` and `EEG/logs`.

## Configure

All machine-specific settings are environment variables (none of them affects results):

```bash
cp .env.example .env    # set PINET_DATA_PATH at least
set -a; source .env; set +a
```

`PINET_CLUSTER=auto` uses slurm when available and local processes otherwise. The grids run
one job per subject (peak memory ~8 GB); results are cached under `PINET_CACHE_PATH`, so
re-running only computes what is missing. `DEVICE` (`Meg` or `Eeg`, default `Meg`) selects the
modality for every script.

## Run

```bash
uv run bash scripts/build_events.sh      # event tables
uv run bash scripts/run_grids.sh         # all grids, MEG and EEG
uv run bash scripts/make_figures.sh      # all figures -> figs/
```

Run `scripts/build_events.sh` before anything else: it caches the event tables (built with
numpy's AVX-512 kernels disabled) that every grid reuses; a grid run first would cache them
without. Then, a quick check that everything is installed (decodes one recording on a 0.1 s window):

```bash
DEVICE=Meg RUN_MODE=test PINET_CLUSTER=none uv run python -m scr.main_hierar
```

Individual steps: `DEVICE=Meg uv run python -m scr.main_hierar`, then
`DEVICE=Meg uv run python -m scr.fig1`.

| Paper figure | Grid | Figure script | Device |
|---|---|---|---|
| Fig 2A-B, hierarchy decoding and half-times | `main_hierar` | `fig1` | Meg |
| Fig S3, EEG replication of Fig 2A-B | `main_hierar` | `fig1_eeg` | Eeg |
| Fig 2D, successive keys (Fig S4: EEG) | `main_fig2_grid` | `fig2` | Meg (Eeg) |
| Fig 3, dynamic code of letters (Fig S5: EEG) | `main_seq` | `fig3-3d` | Meg (Eeg) |
| Fig 4, hierarchy of dynamic codes (Fig S6: EEG) | `main_seq` | `fig4` | Meg (Eeg) |

The script names predate the paper's figure numbering. Figure 1 (design, evoked responses)
and the other supplementary analyses are not part of this repository.

## Notes

- `third_party/neuralset/` is a trimmed copy of the February-2025 [neuralset](https://github.com/facebookresearch/neuroai),
  vendored because the public releases changed API and cache format. Its local changes are
  listed in [third_party/neuralset/README.md](third_party/neuralset/README.md).
- Caching and job submission use [exca](https://github.com/facebookresearch/exca) 0.2.3
  (pinned). Cached results are keyed by configuration, not by code.

## License

- Code: CC BY-NC 4.0, see [LICENSE](LICENSE).
- Vendored neuralset copy: MIT, see the LICENSE file in its directory.
- fastText model (trained on Wikipedia text, hosted on Hugging Face): CC BY-SA 4.0.
- Dataset (hosted on Hugging Face): CC BY-NC 4.0.
