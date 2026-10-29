#!/usr/bin/env bash
# Draw the main figures from the cached grid results into figs/.
#
#   script          paper figure             device
#   scr.fig1        Fig 2A-B                 Meg
#   scr.fig1_eeg    Fig S3                   Eeg
#   scr.fig2        Fig 2D   (Fig S4: EEG)   Meg (Eeg)
#   scr.fig3-3d     Fig 3    (Fig S5: EEG)   Meg (Eeg)
#   scr.fig4        Fig 4    (Fig S6: EEG)   Meg (Eeg)
#
# Every script also runs for the other device; this draws both.
set -euo pipefail
cd "$(dirname "$0")/.."
export RUN_MODE=${RUN_MODE:-all}
for d in Meg Eeg; do
  for fig in fig1 fig1_eeg fig2 fig3-3d fig4; do
    echo "== $fig ($d)"
    DEVICE=$d python -m scr.$fig
  done
done
