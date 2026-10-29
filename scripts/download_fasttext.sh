#!/usr/bin/env bash
# Download the Spanish fastText model used for the syllable embeddings (7.2 GB)
# to scr/model_test/wiki_all_minn1_spanish.bin and check its checksum.
# Model card and training recipe: https://huggingface.co/Lucyzmf/pinet-fasttext-es
set -euo pipefail
cd "$(dirname "$0")/.."
repo=Lucyzmf/pinet-fasttext-es
file=wiki_all_minn1_spanish.bin
sha256=79051c0a7be8262e26553632ab8d4b9e8707c0edb1eb03c0a9275ea13a9f0ac9
mkdir -p scr/model_test
hf download "$repo" "$file" --local-dir scr/model_test
echo "$sha256  scr/model_test/$file" | sha256sum --check
