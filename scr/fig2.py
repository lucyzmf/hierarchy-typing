# %%
import os

if "__file__" in globals():  # run from the project root: figs/ and scr/model_test are relative to it
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator
from mne.stats import fdr_correction
from scipy.stats import wilcoxon
from scr.configs.defaults import workdir
from scr.utils.vis_helper import custom_formatter

fig_path = Path(workdir) / "figs"
fig_path.mkdir(parents=True, exist_ok=True)
# %%


from scr.utils.config import update_config
from scr.configs.defaults import RUN_MODE, default_config, test_config
from scr.configs.word_feat_seq_exp_list import feature_configs, word_seq_configs
from scr.experiments.word_feature_decode_seq_onset import WordFeatureDecodeSeqOnset

from scr.configs.defaults import DEVICE  # env var DEVICE, default Meg

params = {
    "sent_onset": False,
    "study.name": f"Pinet2024{DEVICE}",
    "device.name": DEVICE,
}


scores_dict = {}
# create list of experiments
for seq_config in word_seq_configs:
    for feat_name, feat_conf in feature_configs.items():
        full_config = update_config(default_config, seq_config)
        full_config = update_config(full_config, feat_conf)
        full_config = update_config(full_config, params)
        if RUN_MODE == "test":
            full_config = update_config(full_config, test_config)

        exp = WordFeatureDecodeSeqOnset(**full_config)

        condition = "onset" if full_config["timelock_onset"] else "offset"

        scores = exp.infra.cache_dict
        score = np.array([scores[k] for k in sorted(scores)])  # same subject order across features

        if feat_name not in list(scores_dict.keys()):
            scores_dict[feat_name] = {}
        scores_dict[feat_name][condition] = score


# %%
# for each feature, get the median start of the first second and third event
events = exp.study.build()
# %%


def get_subword_median_start_duration(
    subword_id_col: str = "char_id",
    event_type: str = "Button",
    relative_to: str = "start",
    events: pd.DataFrame = events,
):
    events["start_relative_to_word"] = np.nan
    subwords = events[events.type == event_type]

    if relative_to == "stop":
        subwords["stop"] = subwords["start"] + subwords["duration"]

    group_by_cols = ["subject", "timeline", "sequence_id", "word_id"]

    for i, group in subwords.groupby(group_by_cols):
        if 0 in group[subword_id_col].values:
            word_start = group[group[subword_id_col] == 0].iloc[0][relative_to]

            subwords.loc[group.index, "start_relative_to_word"] = (
                subwords[relative_to] - word_start
            )

    median_duration = subwords.groupby([subword_id_col]).duration.median().reset_index()
    median_start = (
        subwords.groupby([subword_id_col]).start_relative_to_word.median().reset_index()
    )

    if relative_to == "stop":
        median_start["start_relative_to_word"] = -median_start["start_relative_to_word"]
        median_duration["duration"] = -median_duration["duration"]

    return median_start, median_duration


median_start_dict = {}
median_duration_dict = {}

feat_to_event_type_map = {
    "ButtonClass": "Button",
}

for feat in ["ButtonClass"]:
    median_start_dict[feat] = {}
    median_duration_dict[feat] = {}

    for condition in ["onset", "offset"]:
        subword_id_col = "char_id"
        relative_to = "start" if condition == "onset" else "stop"

        median_start, median_duration = get_subword_median_start_duration(
            subword_id_col=subword_id_col,
            event_type=feat_to_event_type_map[feat],
            relative_to=relative_to,
        )

        median_start_dict[feat][condition] = median_start
        median_duration_dict[feat][condition] = median_duration

# %%

plt.rcParams["figure.dpi"] = 500

# get colors
from scr.utils.vis_helper import feat_color

feat_to_title_map = {
    "ButtonClass": "Key",
}


# panel layout in units of the panel's peak mean score, so the x axis, significance
# dots, median boxes and 0-line sit at the same relative height in every panel
Y_LIM = (-0.6, 1.35)  # y limits
DOT_GAP = 0.05  # dot row i at 1 + (i + 2) * DOT_GAP
BOX_TOP, BOX_HEIGHT = -0.3, 0.15  # median boxes, below the x tick labels


def plot_curves(feat, score, ax, condition, normalize=False):
    start_time = -1 if condition == "onset" else -1.5
    duration = 2.5

    n_sub, n_fold, n_shift, n_t = score.shape
    score = score.mean(1)

    ts = np.linspace(start_time, start_time + duration, n_t)

    # plotteps
    base_color = feat_color[feat]
    # Create gradients
    to_white = np.linspace(base_color, [1, 1, 1, 1], 127)  # Gradually add white
    to_black = np.linspace(base_color, [0, 0, 0, 1], 127)  # Gradually subtract color

    # Combine gradients into a single colormap
    custom_cmap = np.vstack([to_white[::-1], to_black])
    custom_colormap = ListedColormap(custom_cmap)
    colors = custom_colormap(np.linspace(0, 1, n_shift + 2))

    if start_time == -1:
        shift_labels = [f"+{i}" for i in range(n_shift + 1)]
        shift_labels[0] = "first"
    else:
        shift_labels = [f"{i+1-n_shift}" for i in range(n_shift)]
        shift_labels[-1] = "last"

    max_mean_score = score.mean(0).max()

    for i in range(n_shift):
        res = wilcoxon(score[:, i], alternative="greater", axis=0)
        _, p = fdr_correction(res.pvalue)
        sig = p < 0.05

        curve = score[:, i].mean(0)
        if normalize:
            curve = (curve - curve.min()) / (curve.max() - curve.min())

        ax.plot(ts, curve, color=colors[i + 1], label=shift_labels[i])
        ax.fill_between(ts, curve, where=sig, alpha=0.3, color=colors[i + 1])

        if normalize:
            ax.scatter(ts, sig * (1 + (i + 2) * 0.05), color=colors[i + 1], s=1)
        else:
            ax.scatter(
                ts,
                sig * max_mean_score * (1 + (i + 2) * DOT_GAP),
                color=colors[i + 1],
                s=1,
            )

    ax.legend(
        loc="upper left",
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(1.05, 1.1),
        labelspacing=0.2,
    )

    sns.despine(ax=ax)
    ax.set_ylim(Y_LIM[0] * max_mean_score, Y_LIM[1] * max_mean_score)

    # plot little squares for median start and duration
    # get median start and duration

    median_start = median_start_dict[feat][condition]
    median_duration = median_duration_dict[feat][condition]

    ordered_shifts = (
        np.arange(n_shift) if condition == "onset" else np.flip(np.arange(n_shift))
    )

    for i in range(n_shift):
        pos_idx = ordered_shifts[i]
        # plot little rectangles for median start and duration
        start = median_start.iloc[pos_idx].start_relative_to_word
        duration = median_duration.iloc[pos_idx].duration

        # plot rectangle below the x tick labels
        rec_height = BOX_HEIGHT * max_mean_score
        y = (BOX_TOP - BOX_HEIGHT) * max_mean_score
        ax.add_patch(
            plt.Rectangle(
                (start, y),
                duration,
                rec_height,
                color=colors[i + 1],
                alpha=0.5,
                edgecolor="none",
            )
        )

    # from the x axis up, clear of the 0 tick label
    ax.vlines(0, 0, Y_LIM[1] * max_mean_score, colors="grey", linestyle="", linewidth=0.7)

    # coarse enough steps (e.g. .01, .05) for custom_formatter's 2 decimals
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
    ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    ax.spines["bottom"].set_position(("data", 0))  # Position at y=0
    ax.xaxis.set_ticks_position("bottom")  # Set ticks on bottom spine
    # y spine and ticks start at the x axis instead of sticking out below it
    y_hi = ax.get_ylim()[1]
    ax.set_yticks([y for y in ax.get_yticks() if 0 <= y <= y_hi])
    ax.spines["left"].set_bounds(0, y_hi)

    xticks = ax.get_xticks()[1:-1]
    xticklabels = [label.get_text() for label in ax.get_xticklabels()][1:-1]

    # Set the new labels to the second subplot
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # mark the window edges (onset 1.5 s, offset -1.5 s) without labelling them
    edges = [x for x in (ts[0], ts[-1]) if x not in xticks]
    ax.set_xticks(np.r_[xticks, edges], xticklabels + [""] * len(edges))

    return ax


# %%
gs_kw = dict(width_ratios=[1, 1], height_ratios=[1], hspace=0.2)
grid = [
    ["button_onset", "button_offset"],
]

all_features = list(scores_dict.keys())

fig, axd = plt.subplot_mosaic(
    grid,
    figsize=(7, 8 / 3),
    gridspec_kw=gs_kw,
    layout="constrained",
)
normalize = False

for k, ax in axd.items():
    f_name, condition = k.split("_")
    full_feat_name = [f for f in all_features if f_name in f.lower()][0]
    feat_score = scores_dict[full_feat_name]

    plot_curves(
        full_feat_name,
        feat_score[condition],
        ax=ax,
        condition=condition,
        normalize=normalize,
    )

axd["button_onset"].set_ylabel("Score")

# Add custom row titles (to the left of the grid)
for i, feat in enumerate(["ButtonClass"]):
    row_title = feat_to_title_map[feat]
    fig.text(
        -0.1,
        1 - (i + 0.5) / 1,
        row_title,
        ha="center",
        va="center",
        fontsize=12,
        color=feat_color[feat],
    )


plt.savefig(fig_path / f"fig2-v3-{DEVICE}.png", bbox_inches="tight")


# %%
