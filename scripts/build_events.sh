#!/usr/bin/env bash
# Build (and cache) the event tables the way the published ones were built: with numpy's
# AVX-512 kernels disabled, for this step only. The behavioural logs contain tied timestamps
# and numpy's default sort orders ties differently on CPUs with AVX-512.
#
# usage: scripts/build_events.sh [Meg] [Eeg]      (default: both)
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PINET_DATA_PATH:?set PINET_DATA_PATH to the dataset root}"
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL"
for d in ${@:-Meg Eeg}; do
  echo "== building $d events (numpy AVX-512 kernels disabled)"
  DEVICE=$d RUN_MODE=all PINET_CLUSTER=none python -c \
    "from scr.main_hierar import exps; df = exps[0].study.build(); print(len(df), 'events')"
done
