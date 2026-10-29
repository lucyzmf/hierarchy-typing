"""
plotting script for Figure 4 (Fig S6 with DEVICE=Eeg)
"""

# %%
import os

if "__file__" in globals():  # run from the project root: figs/ and scr/model_test are relative to it
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

import colorcet
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from mne.stats import fdr_correction
from pydantic import BaseModel, ConfigDict
from scipy.stats import spearmanr, wilcoxon
from scr.configs.defaults import workdir
from scr.experiments.sequential_features import SequentialDecoding
from scr.utils.vis_helper import create_custom_cmap, custom_formatter

fig_path = Path(workdir) / "figs"
fig_path.mkdir(parents=True, exist_ok=True)


# Helpers
class Stats(BaseModel):
    feature: str = "ButtonClass"
    model_config = ConfigDict(extra="forbid")

    def run(self, scores):
        # mean across splits

        if len(scores.shape) == 5:
            scores = scores.mean(2)
        else:
            scores = scores.mean(1)

        shape = scores.shape

        # wilcoxon test
        _, p = wilcoxon(scores.reshape(shape[0], -1), axis=0, alternative="greater")
        p = p.reshape(shape[1:])

        mean = scores.mean(0)
        return np.asarray([mean, p])


# %%
################
# get scores
################
sequential_scoredict = {}
timegen_dict = {}


from scr.utils.config import update_config
from scr.configs.defaults import default_config
from scr.configs.seq_exp_list import seq_config, seq_feature_configs

# which data to use
from scr.configs.defaults import DEVICE  # env var DEVICE, default Meg

study_config = {
    "study.name": f"Pinet2024{DEVICE}",
    "device.name": DEVICE,
    "device.filter": (0.5, 45),
}


# create list of experiments
for feat_name, feat_conf in seq_feature_configs.items():
    full_config = update_config(default_config, study_config)
    full_config = update_config(full_config, seq_config)
    full_config = update_config(full_config, feat_conf)

    exp = SequentialDecoding(**full_config)

    scores = exp.infra.cache_dict
    score = np.array([scores[k] for k in sorted(scores)])  # same subject order across features

    sequential_scoredict[feat_name] = score
    # get center idx
    center_idx = score.shape[1] // 2
    timegen_dict[feat_name] = np.array(score[:, center_idx])

features = list(sequential_scoredict.keys())

subplot_title_mapper = {
    "ContextEmbedding": "Context",
    "SyllableEmbedding": "Syllable",
    "SpacyEmbedding": "Word",
    "ButtonClass": "Letter",
}


# %%
################
# plot
################

# get colors
from scr.utils.vis_helper import feat_color

plt.rcParams["figure.dpi"] = 500
plotting_p = 0.01


def plot_rainbow(f, score, ax):
    means, pvals = Stats(feature=f).run(score)
    n_shifts = len(means)
    center_shift = n_shifts // 2

    times = np.linspace(-1.5, 1.5, means.shape[-1])

    legend_patches = []  # List to hold legend entries

    shifts = np.flip(np.arange(n_shifts))

    # plotteps
    base_color = feat_color[f]
    # Create gradients
    to_white = np.linspace(base_color, [1, 1, 1, 1], 127)  # Gradually add white
    to_black = np.linspace(base_color, [0, 0, 0, 1], 127)  # Gradually subtract color

    # Combine gradients into a single colormap
    custom_cmap = np.vstack([to_black[::-1], to_white])
    custom_colormap = ListedColormap(custom_cmap)
    colors = custom_colormap(np.linspace(0, 1, n_shifts + 2))

    for shift in shifts:
        # left plot
        m = means[shift]
        p = pvals[shift]
        _, p = fdr_correction(p)
        # create gradient colourmap
        gradient_cmap = create_custom_cmap(
            colors[shift + 1], transparencies=[0.0, 0.5, 0.8, 1.0]
        )

        sig = p < 0.05

        if sig.sum() == 0:
            print(f"{f} not showing fold")
            continue

        ax[0].pcolormesh(times, times, m * sig, cmap=gradient_cmap)
        ax[0].contour(
            times,
            times,
            p,
            cmap=gradient_cmap,
            levels=[1e-8, plotting_p],
        )
        ax[0].set_ylim(-1, 1)

        # Create a patch for the legend
        legend_patch = mpatches.Patch(
            color=colors[shift + 1], label=f"P$_{{{center_shift-shift}}}$"
        )
        legend_patches.append(legend_patch)

        # right plot
        ### ax1
        t0_idx = int(1.5 * 50)

        sig_t0 = sig[:, t0_idx]

        # plot test time slice
        ttest0 = m[t0_idx]
        sig_t0 = sig[t0_idx]

        ax[2].plot(times, ttest0, color=colors[shift + 1])
        ax[2].fill_between(
            times, ttest0, where=sig_t0, color=colors[shift + 1], alpha=0.4
        )
        ax[2].set_ylabel("Score")
        if f in ["ButtonClass"]:
            ax[2].set_xlabel("Test Time (s)")

        ymin, ymax = ax[2].get_ylim()

    ax[2].annotate(
        text=subplot_title_mapper[f], color=base_color, xy=(0.7, ymax * 0.8), fontsize=12
    )

    # legend outside axis, remove boarder
    ax[0].legend(
        handles=legend_patches,
        loc="upper left",
        bbox_to_anchor=(1.15, 1),
        frameon=False,
        fontsize=8,
    )

    # formatting
    sns.despine(left=True, bottom=True)
    ax[0].set_ylabel("Train Time (s)")

    # add vertical and horizontal line at t0
    ymin, ymax = ax[0].get_ylim()
    ax[0].vlines(0, ymin, ymax, colors="grey", linewidth=0.5)
    xmin, xmax = ax[0].get_xlim()
    ax[0].hlines(0, xmin, xmax, colors="grey", linewidth=0.5)

    # formatting
    sns.despine(left=True, bottom=True)
    if f in ["ButtonClass"]:
        ax[0].set_xlabel("Test Time (s)")

    ax[0].set_ylabel("Train Time (s)")

    if f not in ["ButtonClass"]:
        ax[0].xaxis.set_visible(False)

    # add vertical and horizontal line at t0
    ymin, ymax = ax[0].get_ylim()
    ax[0].vlines(0, ymin, ymax, colors="grey", linewidth=0.5)
    xmin, xmax = ax[0].get_xlim()
    ax[0].hlines(0, xmin, xmax, colors="grey", linewidth=0.5)

    ax[0].yaxis.set_major_formatter(FuncFormatter(custom_formatter))
    ax[0].xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    ax[2].xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    ax[2].yaxis.set_major_formatter(FuncFormatter(custom_formatter))


def plot_timegen(f, score, ax):
    times = np.linspace(-1.5, 1.5, score.shape[-1])
    sel = np.logical_and(times > -1, times < 1.5)
    score = score[:, :, :, sel][:, :, sel, :]
    means, pvals = Stats(feature=f).run(score)
    _, p = fdr_correction(pvals)

    color = feat_color[f]
    gradient_cmap = create_custom_cmap(color, transparencies=[0.0, 0.5, 0.8, 0.9, 1.0])

    times = np.linspace(-1, 1.5, means.shape[-1])

    sig = p < 0.05

    ax.pcolormesh(times, times, means * sig, cmap=gradient_cmap)

    levels = [1e-8, plotting_p]
    ax.contour(times, times, p, cmap=gradient_cmap, levels=levels)

    # remove all spines
    sns.despine(left=True, bottom=True)
    ax.set_ylabel("Train Time (s)")
    ax.set_xlabel("Test Time (s)")

    # plot vertical and horizontal lines at t0
    ymin, ymax = ax.get_ylim()
    ax.vlines(0, ymin, ymax, colors="grey", linestyle="--")
    ax.hlines(0, ymin, ymax, colors="grey", linestyle="--")
    ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))


def get_midpoint_t(t, scores, is_onset):
    t0 = int(scores.shape[-1] / 2)
    freq = int(exp.device.frequency)
    if is_onset:
        scores_ = scores[t0 - freq : t0]
        t_ = t[t0 - freq : t0]

    else:
        scores_ = scores[t0 : t0 + freq]
        t_ = t[t0 : t0 + freq]

    half = (scores_.max()) / 2
    diff = np.abs(scores_ - half)
    half_idx = np.argmin(diff)

    return float(t_[half_idx])


def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min())


# %%
right = [
    ["context_seq"],
    ["spacy_seq"],
    ["syllable_seq"],
    ["button_seq"],
]

mid = [
    ["context_slice"],
    ["spacy_slice"],
    ["syllable_slice"],
    ["button_slice"],
]

left = [
    ["gen", "gen", "gen", "gen"],
    ["gen", "gen", "gen", "gen"],
    ["gen", "gen", "gen", "gen"],
    ["width", "width", "width", "width"],
]


gs_kw = dict(width_ratios=[1, 0.7, 0.7], hspace=1, wspace=1)
fig, axd = plt.subplot_mosaic(
    [[left, mid, right]],
    gridspec_kw=gs_kw,
    figsize=(12, 6),
    layout="constrained",
)

for k, ax in axd.items():
    if "seq" not in k:
        continue

    f = k.split("_")[0]
    feat_name = [feat for feat in features if f in feat.lower()][0]

    plot_rainbow(
        feat_name, sequential_scoredict[feat_name], [ax, None, axd[f"{f}_slice"]]
    )

for i in range(len(features)):
    plot_timegen(features[i], timegen_dict[features[i]], axd["gen"])
axd["gen"].set_xlim(-1, 1.5)
axd["gen"].set_ylim(-1, 1.5)


label_fontsize = 12
axd["context_seq"].text(
    -0.5,
    1.1,
    "d",
    transform=axd["context_seq"].transAxes,
    va="top",
    fontsize=label_fontsize,
)
axd["gen"].text(
    -0.2, 1.0, "a", transform=axd["gen"].transAxes, va="top", fontsize=label_fontsize
)

axd["context_slice"].text(
    -0.2,
    1,
    "c",
    transform=axd["context_slice"].transAxes,
    va="top",
    fontsize=label_fontsize,
)


# plot horizontal width of generalisation of timegens
df_halftime = pd.DataFrame()
ts = np.arange(-1.5, 1.5, 1 / exp.device.frequency)
scores = []

t_sel = 1.5

for i, (feat_name, score) in enumerate(timegen_dict.items()):
    score = score.mean(1)

    n_subj, n_t, _ = score.shape

    # select t0
    t0_idx = int(t_sel * 50)
    score = score[:, t0_idx]

    axd["width"].plot(
        ts, min_max_norm(score.mean(0)), label=feat_name, color=feat_color[feat_name]
    )

    half_times_on = [get_midpoint_t(ts, score[i], is_onset=True) for i in range(n_subj)]
    half_times_off = [get_midpoint_t(ts, score[i], is_onset=False) for i in range(n_subj)]

    df = pd.DataFrame(
        {
            "subject": np.arange(len(half_times_on)),
            "half_time_dur": np.array(half_times_off) - np.array(half_times_on),
            "feature": feat_name,
        }
    )
    df_halftime = pd.concat([df_halftime, df])

    res = wilcoxon(score, axis=0, alternative="greater")
    _, p_corrected = fdr_correction(res.pvalue)

    axd["width"].scatter(
        ts[p_corrected < 0.05],
        np.full((p_corrected < 0.05).sum(), 1.2 - i * 0.05),
        color=feat_color[feat_name],
        s=2,
    )

    # duration of significance
    t_first = ts[np.where(p_corrected < 0.05)[0][0]]
    t_last = ts[np.where(p_corrected < 0.05)[0][-1]]
    print(f"sig duration for feature {feat_name} {t_last-t_first:.3f}")

axd["width"].set_xlabel("Test Time (s)")
axd["width"].set_xlim(-1, 1)
axd["width"].set_ylabel("Score$_n$")
axd["width"].annotate("D$_{{0\\mathrm{{ms}}}}$", xy=(0.5, 0.5), fontsize=12)
axd["width"].xaxis.set_major_formatter(FuncFormatter(custom_formatter))
axd["width"].yaxis.set_major_formatter(FuncFormatter(custom_formatter))
sns.despine(ax=axd["width"])

import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# add colorbar for seq
grey_cmap = colorcet.cm.gray_r
norm = mpl.colors.Normalize(vmin=0, vmax=1)

cax = inset_axes(
    axd["button_seq"],
    width="3%",
    height="20%",
    loc="lower right",
    bbox_to_anchor=axd["button_seq"].bbox,
)
fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=grey_cmap),
    cax=cax,
    orientation="vertical",
).set_label("Score$_n$", fontsize=8)

# repeat for timegen
cax = inset_axes(
    axd["gen"],
    width="20%",
    height="15%",
    loc="lower right",
    bbox_to_anchor=(0.8, 0.2, 0.1, 1),  # Move left by reducing x value (first parameter)
    bbox_transform=axd["gen"].transAxes,
)
fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=grey_cmap),
    cax=cax,
    orientation="vertical",
    label="Score$_n$",
)

plt.savefig(fig_path / f"fig4-{DEVICE}-v2.png", bbox_inches="tight")


# %%
sns.boxplot(
    data=df_halftime,
    x="half_time_dur",
    y="feature",
    hue="feature",
    palette=feat_color,
    showfliers=False,
)
plt.show()

# pair wise significance
for feat1, feat2 in zip(features[:-1], features[1:]):
    res = wilcoxon(
        df_halftime.query("feature==@feat1").half_time_dur,
        df_halftime.query("feature==@feat2").half_time_dur,
    )
    print(f"{feat1} vs {feat2}, p={res.pvalue}")


def get_r(group):
    cat_order = pd.CategoricalDtype(features)
    group["feature"] = group["feature"].astype(cat_order)
    group = group.sort_values(by="feature")
    return spearmanr(
        np.flip(group["feature"].cat.codes), group["half_time_dur"]
    ).statistic


# corr per subj
df_corr = df_halftime.groupby("subject").apply(get_r).reset_index()
df_corr.rename(columns={0: "r"}, inplace=True)
df_corr

rs = df_corr["r"].values

res = wilcoxon(rs, alternative="greater")
print(
    f"corr between half time duration and feature order mean {rs.mean()}, sem {rs.std() / np.sqrt(len(rs))}, p={res.pvalue}"
)

# %%
# find peak and first index sig for decoding curves

for feat_name in timegen_dict.keys():
    score = timegen_dict[feat_name].mean(1)
    _, p = wilcoxon(score, axis=0, alternative="greater")
    _, sig = fdr_correction(p)

    diag_score = np.diagonal(score, axis1=-1, axis2=-2)
    sig = np.diagonal(sig, axis1=-2, axis2=-1)

    means = diag_score.mean((0))
    times = np.arange(-1.5, 1.5, 1 / exp.device.frequency)

    plt.plot(times, means, label=feat_name, color=feat_color[feat_name])

    plt.fill_between(
        times, means, alpha=0.2, where=sig < 0.05, color=feat_color[feat_name]
    )

    print(f"{feat_name}")
    print(f"peak t {times[np.argmax(means)]:.3f}")
    print(f"peak score {np.max(means):.3f}")


    sig_index = np.where(sig < 0.05)[0]
    # find first three consecutive sig
    for i in range(len(sig_index) - 2):
        if sig_index[i + 2] - sig_index[i] == 2:
            print(f"first sig {times[sig_index[i]]:.3f}")
            break

    # find last three consecutive sig
    for i in range(len(sig_index) - 2, 0, -1):
        if sig_index[i] - sig_index[i - 2] == 2:
            print(f"last sig {times[sig_index[i]]:.3f}")
            break

    print("")

plt.legend()
plt.show()


# %%
# get half times
def get_midpoint_t(t, scores):
    t0 = int(-exp.start * exp.device.frequency)
    scores_ = scores[int(t0 - exp.device.frequency) : t0]
    t_ = t[int(t0 - exp.device.frequency) : t0]

    half = (scores_.max()) / 2
    diff = np.abs(scores_ - half)
    half_idx = np.argmin(diff)

    return float(t_[half_idx])


df_halftime = pd.DataFrame()

for feat_name in timegen_dict.keys():
    score = timegen_dict[feat_name].mean(1)

    diag_score = np.diagonal(score, axis1=-2, axis2=-1)

    ts = np.arange(-1.5, 1.5, 1 / exp.device.frequency)

    half_times = [get_midpoint_t(ts, diag_score[i]) for i in range(diag_score.shape[0])]
    df = pd.DataFrame(
        {
            "half_time": half_times,
            "feature": feat_name,
            "subject": np.arange(len(half_times)),
        }
    )
    df_halftime = pd.concat([df_halftime, df])

fig = plt.figure(figsize=(6, 3))
sns.boxplot(
    data=df_halftime,
    x="half_time",
    y="feature",
    hue="feature",
    palette=feat_color,
    orient="h",
)
sns.despine()
plt.show()

# %%
# significance testing
for feat1, feat2 in zip(features[:-1], features[1:]):
    res = wilcoxon(
        df_halftime.query("feature==@feat1").half_time,
        df_halftime.query("feature==@feat2").half_time,
    )
    print(f"{feat1} vs {feat2}, p={res.pvalue}")

# %%
# quantify width of timegens
df_realign_to_diag = pd.DataFrame()

# compute dist to diag once
n_t = int(exp.duration * exp.device.frequency)
dist_to_diag = np.zeros((n_t, n_t))
for i in range(n_t):
    for j in range(n_t):
        dist_to_diag[i, j] = (i - j) / np.sqrt(2)

for feat_name, score in timegen_dict.items():
    score = score.mean(1)

    n_subj, n_t, _ = score.shape

    xi, yi = np.meshgrid(np.arange(n_t), np.arange(n_t))

    score = score.reshape(n_subj, -1)

    res = wilcoxon(score, axis=0, alternative="greater")
    _, p_corrected = fdr_correction(res.pvalue)

    for s in range(n_subj):
        temp_df = pd.DataFrame(
            {
                "dist_to_diag": dist_to_diag.ravel(),
                "t": dist_to_diag.ravel() / exp.device.frequency,
                "score": score[s],
                "feature": feat_name,
                "subject": s,
            }
        )
        df_realign_to_diag = pd.concat([df_realign_to_diag, temp_df])

sns.lineplot(data=df_realign_to_diag, x="t", y="score", hue="feature")
plt.show()
# %%
# keep -1 s < t < 1 s
df_realign_to_diag = df_realign_to_diag.query("t < 1 and t > -1")
df_realign_to_diag.reset_index(drop=True, inplace=True)

sns.lineplot(data=df_realign_to_diag, x="t", y="score", hue="feature", palette=feat_color)
plt.show()

# %%
# quantify horizontal width
for feat_name, score in timegen_dict.items():
    score = score.mean(1)

    n_subj, n_t, _ = score.shape

    res = wilcoxon(score.reshape(n_subj, -1), axis=0, alternative="greater")
    _, p_corrected = fdr_correction(res.pvalue)
    p_corrected = p_corrected.reshape(n_t, n_t)

    n_nonzero_rows = (np.sum(p_corrected < 0.05, axis=1) > 10).sum()

    print(f"{feat_name} has {n_nonzero_rows} rows with significant scores")

    # avg generalisation
    sig = p_corrected < 0.05
    print(
        f"{feat_name} has avg generalisation {sig.sum()/n_nonzero_rows/exp.device.frequency:.3f}"
    )


# %%
