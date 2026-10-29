#!/usr/bin/env bash
# Run the decoding grids behind the main figures, for MEG and EEG.
#
#   scr.main_hierar      paper Fig 2A-B / S3: hierarchy of Context / Word / Syllable / Letter
#   scr.main_fig2_grid   paper Fig 2D / S4:   successive keys
#   scr.main_seq         paper Fig 3-4 / S5-S6: temporal generalisation, dynamic codes
#
# Results are cached under PINET_CACHE_PATH; re-running only computes what is missing.
# Build the event tables first (scripts/build_events.sh).
#
# usage: scripts/run_grids.sh [Meg] [Eeg]      (default: both)
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PINET_DATA_PATH:?set PINET_DATA_PATH to the dataset root}"
for d in ${@:-Meg Eeg}; do
  for grid in main_hierar main_fig2_grid main_seq; do
    echo "== $grid ($d)"
    DEVICE=$d python -m scr.$grid
  done
done
