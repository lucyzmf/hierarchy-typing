"""Plot Fig 2A-B (hierarchy decoding and half-times).

Panel A shows the group-mean decoding curve of each feature divided by its maximum
within the half-time window ([-1, 0) s for onset, [0, 1) s for offset), drawn over
that window only. Half-times and statistics are computed per participant.
"""

# %%
import os

if "__file__" in globals():  # run from the project root: figs/ and scr/model_test are relative to it
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import typing as tp
from pathlib import Path

import colorcet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scr.utils.config import update_config
from pydantic import BaseModel
from scipy.stats import wilcoxon
from scr.configs.defaults import default_config, workdir
from scr.configs.hierarchy_exp_list import hierarchy_configs, hierarchy_feature_configs
from scr.experiments.word_feature_decode import WordFeatureDecode
from scr.utils.vis_helper import custom_formatter

fig_path = Path(workdir) / "figs"
fig_path.mkdir(parents=True, exist_ok=True)


def get_midpoint_t(t, scores, is_onset):
    if is_onset:
        t0 = int(scores.shape[-1] * 0.75)
        scores_ = scores[int(t0 / 3) : t0]
        t_ = t[int(t0 / 3) : t0]

    else:
        t0 = int(scores.shape[-1] * 0.25)
        scores_ = scores[t0 : int(t0 * 3)]
        t_ = t[t0 : int(t0 * 3)]

    half = (scores_.max()) / 2
    diff = np.abs(scores_ - half)
    half_idx = np.argmin(diff)

    return float(t_[half_idx])


def mean_max_normalized_summary(scores, is_onset):
    """Average across participants and folds, then divide by the maximum within the
    half-time window (same window as get_midpoint_t). Returns (window, normalized)."""
    mean_scores = np.asarray(scores, dtype=float).mean(axis=(0, 1))
    if is_onset:
        t0 = int(mean_scores.shape[-1] * 0.75)
        window = slice(int(t0 / 3), t0)
    else:
        t0 = int(mean_scores.shape[-1] * 0.25)
        window = slice(t0, int(t0 * 3))

    window_scores = mean_scores[window]
    if window_scores.max() <= 0:
        raise ValueError("Window maximum must be positive")
    return window, window_scores / window_scores.max()


def set_time_axis(ax, condition):
    """Match the displayed axes to the exact half-time analysis windows."""
    if condition == "onset":
        ax.set_xlim(-1.0, 0.0)
        ticks = [-1.0, -0.75, -0.5, -0.25, 0.0]
    else:
        ax.set_xlim(0.0, 1.0)
        ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    labels = [custom_formatter(x, None) for x in ticks]
    labels[ticks.index(0.0)] = condition
    if condition == "onset":
        labels[0] = "-1 s"
    else:
        labels[-1] = "1 s"
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


class Stats(BaseModel):
    feature: str
    score: tp.Any
    t_displayed: tp.Any
    _mean: tp.Any = None
    _median: tp.Any = None
    _sig: tp.Any = None

    def model_post_init(self, log__: tp.Any) -> None:
        # get mean to plot
        self._mean = self.score.mean((0, 1))
        self._median = np.median(self.score.mean(1), 0)

    def get_ps(self):
        n_subj, n_fold, n_times = self.score.shape

        from mne.stats import fdr_correction
        from scipy.stats import wilcoxon

        assert len(self.score.shape) == 3, "need to be 3D array, shape subj fold time"

        # check if there are all zeros in the score if so create mask before wilcoxon
        non_zero_cols = self.score.mean(1).sum(0) != 0
        pval = np.zeros(n_times)

        res = wilcoxon(self.score.mean(1)[:, non_zero_cols], axis=0)
        pval[non_zero_cols] = res.pvalue
        rejected, p_corrected = fdr_correction(pval)

        return p_corrected


# %%
######################
# get cached results
######################
hierar_dict = {}
half_times = {}

from scr.configs.defaults import DEVICE  # env var DEVICE, default Meg

study_config = {
    "study.name": f"Pinet2024{DEVICE}",
    "device.name": DEVICE,
    "group_by": "sequence_id",
}

# create list of exp for hierarchy analysis
for i, h_conf in enumerate(hierarchy_configs):
    for feat_name, feat_conf in hierarchy_feature_configs.items():
        full_config = update_config(default_config, h_conf)
        full_config = update_config(full_config, feat_conf)
        full_config = update_config(full_config, study_config)

        exp = WordFeatureDecode(**full_config)
        scores = exp.infra.cache_dict

        # Fail fast with an actionable message if the cache is empty for this
        # (device, feature, onset) combination, instead of letting an empty
        # 1D array propagate into Stats.model_post_init and crash inside
        # pydantic with the opaque "axis 1 is out of bounds" error.
        if len(scores) == 0:
            raise RuntimeError(
                f"Empty cache for feature={feat_name}, "
                f"study={full_config['study']['name']}, "
                f"device={full_config['device']['name']}, "
                f"onset={exp.onset} at "
                f"{exp.infra.uid_folder()}. "
                "Run main_hierar.py (with the matching DEVICE env var) to "
                "populate the cache before plotting."
            )

        if (
            ("Syllable" in feat_name) or ("Button" in feat_name)
        ) and exp.subscore_func != "null":
            scores = np.array([scores[k][:, 0] for k in sorted(scores)])
        else:
            scores = np.array([scores[k] for k in sorted(scores)])  # same subject order across features


        t = np.arange(exp.start, exp.start + exp.duration, step=1 / exp.device.frequency)

        # Ensure feat_name_full exists in hierar_dict
        if feat_name not in hierar_dict:
            hierar_dict[feat_name] = {}

        hierar_dict[feat_name]["onset" if exp.onset else "offset"] = Stats(
            feature=feat_name,
            score=scores,
            t_displayed=t,
        )

        # get half time
        key_onset = "onset" if exp.onset else "offset"
        if key_onset not in half_times:
            half_times[key_onset] = {}
        half_times[key_onset][feat_name] = [
            get_midpoint_t(t, scores[s].mean(0), exp.onset) for s in range(len(scores))
        ]




# %%
# reorder
ordered_feat = [
    "ContextEmbedding",
    "SpacyEmbedding",
    "SyllableEmbedding",
    "ButtonClass",
]

ordered_y_label = ["r" if "Button" not in f else "acc" for f in ordered_feat]

cmap = colorcet.cm.rainbow_r

feat_color = {}

for f in range(len(ordered_feat)):
    feat_color[ordered_feat[f]] = cmap(f / (len(ordered_feat) - 1))


# %%
plt.rcParams["figure.dpi"] = 500

left = [
    ["context_onset", "context_offset"],
    ["spacy_onset", "spacy_offset"],
    ["syllable_onset", "syllable_offset"],
    ["button_onset", "button_offset"],
]

right = [
    ["on", "off"],
    ["on", "off"],
    ["on", "off"],
    ["time_l", "time_r"],
    ["time_l", "time_r"],  # behavioural panels
]

gs_kw = dict(width_ratios=[1.8, 2.2])
fig, axd = plt.subplot_mosaic(
    [
        [left, right],
    ],
    gridspec_kw=gs_kw,
    figsize=(12, 5),
    layout="constrained",
)

k_to_legend = {
    "context_onset": "Context",
    "spacy_onset": "Word",
    "syllable_onset": "Syllable",
    "button_onset": "Key",
}

feat_to_legend = {
    "ContextEmbedding": "Context",
    "SpacyEmbedding": "Word",
    "SyllableEmbedding": "Syllable",
    "ButtonClass": "Letter",
}

for f, ax in axd.items():
    sns.despine(ax=ax)
    if "set" not in f:
        continue

    # get scores
    feat, condition = f.split("_")

    feat_name_full = [f for f in hierar_dict.keys() if feat in f.lower()][0]
    stat = hierar_dict[feat_name_full][condition]
    sig = stat.get_ps()

    # plot
    ax.plot(stat.t_displayed, stat._mean, color=feat_color[feat_name_full])
    ax.fill_between(
        stat.t_displayed,
        stat._mean,
        where=(sig < 0.05),
        color=feat_color[feat_name_full],
        alpha=0.6,
    )
    ax.spines["bottom"].set_position(("data", 0))
    ax.tick_params(axis="y", direction="in")

    if condition == "onset":
        ax.set_ylabel(
            ordered_y_label[ordered_feat.index(feat_name_full)], rotation=0, labelpad=10
        )
        # annotate
        ax.annotate(
            k_to_legend.get(f),
            xy=(-1.25, stat._mean.max() / 2),
            fontsize=14,
            color=feat_color[feat_name_full],
        )
        ax.set_xlim(-1.5, 0.5)

        # format xlabeltics
        ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))

    else:
        ax.set_xlim(-0.5, 1.5)

        ymin, ymax = axd[f"{feat}_offset"].get_ylim()
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_visible(False)

    # remove xticks
    if "button" not in f:
        ax.xaxis.set_visible(False)
    else:
        ax.axhline(y=0, color="black", linewidth=1)

        # change tick label
        # Get the current x-ticks and x-tick labels for the second subplot
        ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))

        xticks = ax.get_xticks()
        xticklabels = [label.get_text() for label in ax.get_xticklabels()]

        # Find the index of the x-tick closest to 5
        labels_sel = [x in ["0", "0.0"] for x in xticklabels]
        idx = [index for index, value in enumerate(labels_sel) if value][0]

        # Replace the specific tick label
        xticklabels[idx] = condition
        if condition == "onset":
            xticklabels[0] = "-1.5 s"
        else:
            xticklabels[-1] = "1.5 s"

        # Set the new labels to the second subplot
        ax.set_xticklabels(xticklabels)

        # label onset and offset
    ymin, ymax = ax.get_ylim()
    ax.vlines(0, ymin, ymax, colors="grey", linestyle="", linewidth=0.7)

# add white astrik annotation to context
axd["context_onset"].annotate("*", xy=(0.08, 0.07), fontsize=20, color="white")

# group mean / window maximum, over the half-time window
for f in ["on", "off"]:
    ax = axd[f]
    condition = "onset" if f == "on" else "offset"

    for i in range(len(ordered_feat)):
        feat = ordered_feat[i]
        stat = hierar_dict[feat][condition]
        window, normalised_score = mean_max_normalized_summary(
            stat.score, condition == "onset"
        )
        displayed_times = stat.t_displayed[window]

        ax.plot(displayed_times, normalised_score, color=feat_color[feat], label=feat)
        ax.annotate(
            feat_to_legend[feat],
            xy=(-0.75, 0.8 - i * 0.1),
            fontsize=14,
            color=feat_color[feat],
        )

    ax.tick_params(axis="y", direction="in")
    ax.spines["bottom"].set_position(("data", 0))

    if condition == "offset":
        # remove yticks
        ax.set_yticks([])
    else:
        ax.set_ylabel("Score$_{n}$")

    ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
    set_time_axis(ax, condition)

# add half time
data = []
# Loop through each key and corresponding values in the dictionary
for is_onset, features in half_times.items():
    for feature_name, half_times_ in features.items():
        for half_time in range(len(half_times_)):
            # Append each feature, its half time, and its type to the list
            data.append(
                {
                    "feature": feature_name,
                    "half_time": half_times_[half_time],
                    "is_onset": is_onset == "onset",  # True if 'onset', False if 'offset'
                    "subject": half_time,
                }
            )

# Convert the list of dictionaries to a DataFrame
df_half_times = pd.DataFrame(data)

for f in ["time_l", "time_r"]:
    ax = axd[f]
    onset = "l" in f  # True if "l" is in k, False otherwise

    sns.boxplot(
        data=df_half_times.query("is_onset == @onset"),
        y="feature",
        x="half_time",
        ax=ax,
        palette=feat_color,
        order=ordered_feat,
        width=0.6,
        showcaps=False,
        flierprops={"marker": " "},
    )

    sns.despine(ax=ax, left=True)
    ax.yaxis.set_visible(False)
    ax.set_xlabel("")
    set_time_axis(ax, "onset" if onset else "offset")

    if onset:
        ax.annotate("Half Times", xy=(-1.0, 3.0), fontsize=14, color="black")


# Add significance brackets
def add_bracket(ax, y1, y2, x, h, text, text_offset=0.05):
    """
    Adds a significance bracket between two groups on a plot.

    Args:
        ax: The Matplotlib Axes object.
        x1, x2: Positions of the two groups on the x-axis.
        y: The y-coordinate of the bracket.
        h: The height of the bracket.
        text: The significance label (e.g., '*', '**', 'p<0.05').
    """
    ax.plot([x, x + h, x + h, x], [y1, y1, y2, y2], color="black", lw=1.0)
    ax.text(
        x + h + text_offset, (y1 + y2) / 2, text, ha="center", va="center", fontsize=10
    )


# get pairwise comparison p
# check between each group whether half time is significantly different
df_half_times_subset = df_half_times.query("feature in @ordered_feat")

onset_p = []
offset_p = []

for is_onset in [True]:
    df_onset_halftimes_temp = df_half_times_subset[
        df_half_times_subset.is_onset == is_onset
    ]

    for feat1, feat2 in zip(ordered_feat[:-1], ordered_feat[1:]):
        res = wilcoxon(
            df_onset_halftimes_temp.query("feature==@feat1").half_time,
            df_onset_halftimes_temp.query("feature==@feat2").half_time,
        )
        print(f"is onset {is_onset}: {feat1} vs {feat2}, p={res.pvalue}")
        if is_onset:
            onset_p.append(res.pvalue)
        else:
            offset_p.append(res.pvalue)

df_offset_halftimes_temp = df_half_times_subset[df_half_times_subset.is_onset == False]
# all pair wise comparison
for i in range(len(ordered_feat)):
    for j in range(i + 1, len(ordered_feat)):
        res = wilcoxon(
            df_offset_halftimes_temp.query("feature==@ordered_feat[@i]").half_time,
            df_offset_halftimes_temp.query("feature==@ordered_feat[@j]").half_time,
        )
        print(f"is onset False: {ordered_feat[i]} vs {ordered_feat[j]}, p={res.pvalue}")

        if res.pvalue < 0.05:
            label = "**" if res.pvalue < 0.01 else "*"
            add_bracket(
                axd["time_r"], i, j, 0.6 + 0.08 * (i + j), 0.02, label, text_offset=-0.03
            )

# add significance brackets
for i, p in enumerate(onset_p):
    if p < 0.05:
        label = "**" if p < 0.01 else "*"
        add_bracket(
            axd["time_l"], i, i + 1, -0.9 + 0.2 * i, -0.02, label, text_offset=0.03
        )


# add subfig labels at upper left corner of each subfig
label_fontsize = 12
axd["context_onset"].text(
    -0.2,
    1.4,
    "a",
    transform=axd["context_onset"].transAxes,
    fontsize=label_fontsize,
    va="top",
)
axd["on"].set_ylim(0, 1.1)
axd["off"].set_ylim(0, 1.1)

plt.savefig(
    fig_path
    / f"fig1-v3-{full_config['device']['name']}_subscore{exp.subscore_func}.png",
    bbox_inches="tight",
)
# %%
# find peak and first index sig for decoding curves

for feat_name in hierar_dict.keys():
    for condition in ["onset", "offset"]:
        stat = hierar_dict[feat_name][condition]
        n_subj = stat.score.shape[0]
        sig = stat.get_ps()
        print(f"{feat_name} {condition}")
        print(
            f"peak {stat.t_displayed[np.argmax(stat._mean)]:.3f} with score {np.max(stat._mean):.3f}, sem {np.std(stat.score[:, :, np.argmax(stat._mean)].mean(1))/np.sqrt(n_subj):.3f}"
        )
        if condition == "onset":
            sig_index = np.where(sig < 0.05)[0]
            # find first three consecutive sig
            for i in range(len(sig_index) - 2):
                if sig_index[i + 2] - sig_index[i] == 2:
                    print(f"first sig {stat.t_displayed[sig_index[i]]:.3f}")
                    break
        else:
            # find last three consecutive sig
            sig_index = np.where(sig < 0.05)[0]
            for i in range(len(sig_index) - 2, 0, -1):
                if sig_index[i] - sig_index[i - 2] == 2:
                    print(f"last sig {stat.t_displayed[sig_index[i]]:.3f}")
                    break
        print("")
# %%
# get correlation of half times
# for each subj, compute corr of half time with feature order
from scipy.stats import spearmanr


def get_r(group):
    cat_order = pd.CategoricalDtype(ordered_feat)
    group["feature"] = group["feature"].astype(cat_order)
    group = group.sort_values(by="feature")
    return spearmanr(np.flip(group["feature"].cat.codes), group["half_time"]).statistic


df_corr = df_half_times.groupby(["subject", "is_onset"]).apply(get_r).reset_index()
df_corr.rename(columns={0: "r"}, inplace=True)

# sig test
rs_onset = df_corr.query("is_onset").r.values

res = wilcoxon(rs_onset, alternative="less")
print(
    f"corr between onset and feature order: {rs_onset.mean():.2f}, sem {rs_onset.std()/np.sqrt(len(rs_onset))}, p={res.pvalue}"
)

rs_offset = df_corr.query("not is_onset").r.values

res = wilcoxon(rs_offset, alternative="greater")
print(
    f"corr between offset and feature order: {rs_offset.mean():.2f}, sem {rs_offset.std()/np.sqrt(len(rs_offset))}, p={res.pvalue}"
)

# %%
