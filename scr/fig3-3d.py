"""
plotting script for figure 3 focus on button
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
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from mne.stats import fdr_correction
from scr.utils.config import update_config
from pydantic import BaseModel, ConfigDict
from scipy.stats import wilcoxon
from scr.configs.defaults import default_config, workdir
from scr.configs.seq_exp_list import seq_config, seq_feature_configs
from scr.experiments.sequential_features import SequentialDecoding
from scr.utils.vis_helper import create_custom_cmap, custom_formatter

fig_path = Path(workdir) / "figs"
fig_path.mkdir(parents=True, exist_ok=True)

plotting_p = 0.01
move = 0.0


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

# which data to use
from scr.configs.defaults import DEVICE  # env var DEVICE, default Meg

study_config = {
    "study.name": f"Pinet2024{DEVICE}",
    "device.name": DEVICE,
    "device.filter": (0.5, 45),
}

feat_name = "ButtonClass"
feat_conf = seq_feature_configs[feat_name]

feat_name_display_dict = {
    "ContextEmbedding": "Context",
    "SyllableEmbedding": "Syllable",
    "SpacyEmbedding": "Word",
    "ButtonClass": "Letter",
}

# create list of experiments
full_config = update_config(default_config, study_config)
full_config = update_config(full_config, seq_config)
full_config = update_config(full_config, feat_conf)

exp = SequentialDecoding(**full_config)

scores = exp.infra.cache_dict
sequential_score = np.array([scores[k] for k in sorted(scores)])  # deterministic subject order

center_idx = sequential_score.shape[1] // 2
timegen_score = np.array(sequential_score[:, center_idx])


# %%
################
# plot
################

plt.rcParams["figure.dpi"] = 500

# get colors
from scr.utils.vis_helper import feat_color

base_color = feat_color[feat_name]
# Create gradients
to_white = np.linspace(base_color, [1, 1, 1, 1], 127)  # Gradually add white
to_black = np.linspace(base_color, [0, 0, 0, 1], 127)  # Gradually subtract color

# Combine gradients into a single colormap
custom_cmap = np.vstack([to_black[::-1], to_white])
custom_colormap = ListedColormap(custom_cmap)
colors = custom_colormap(np.linspace(0, 1, exp.shifts * 2 + 1 + 2))


import xarray as xr


def get_data(data=sequential_score):
    n_subjects, n_positions, n_splits, n_traintimes, n_testtimes = data.shape
    times = np.linspace(-1.5, 1.5, n_traintimes)
    shifts = np.arange(n_positions) - n_positions // 2
    return xr.DataArray(
        data,
        dims=["subject", "position", "split", "train_time", "test_time"],
        coords={
            "subject": range(n_subjects),
            "position": shifts,
            "split": range(n_splits),
            "train_time": times,
            "test_time": times,
        },
    )


from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def plot_fading(ax, xyz, color="k", **kwargs):

    # trailing transparency
    alphas = np.logspace(-5, 0, xyz.shape[1])

    # segment for each level of transparency
    segments = np.array([xyz[:, i : i + 3].T for i in range(len(alphas) - 2)])

    colors = color[None] * np.ones_like(alphas)[:, None]
    colors[:, 3] = alphas

    # plot lines
    Line = Line3DCollection if len(xyz) == 3 else LineCollection
    lc = Line(segments, color=colors, **kwargs)
    ax.add_collection(lc)

    # plot final marker
    s = max(10, 10 + (xyz[:, -1]).sum(0) * 20)
    ax.scatter(*xyz[:, -1], color=colors[-1, :3], s=s, edgecolor="k", linewidth=0.5)


def plot_rainbow(f, score, ax):
    means, pvals = Stats(feature=f).run(score)
    n_shifts = len(means)

    times = np.linspace(-1.5, 1.5, means.shape[-1])

    legend_patches = []  # List to hold legend entries

    shifts = np.flip(np.arange(n_shifts))

    for shift in shifts:
        # left plot
        m = means[shift]
        p = pvals[shift]
        _, p = fdr_correction(p)
        # create gradient colourmap
        gradient_cmap = create_custom_cmap(colors[shift + 1], transparencies=[0.0, 0.3])

        sig = p < 0.05

        if sig.sum() == 0:
            print(f"{f} not showing fold")
            continue

        ax[0].pcolormesh(times + (4 - shift) * move, times, sig, cmap=gradient_cmap)
        contour_cmap = create_custom_cmap(colors[shift + 1], transparencies=[1, 1])
        ax[0].contour(
            times + (4 - shift) * move,
            times,
            p,
            cmap=contour_cmap,
            levels=[1e-8, plotting_p],
        )
        ax[0].set_ylim(-1, 1)

        # Create a patch for the legend
        shift_label = 4 - shift
        if shift_label < 0:
            legend_patch = mpatches.Patch(
                color=colors[shift + 1],
                label=f"{feat_name_display_dict[f]}$_\\minus$$_{-shift_label}$",
            )
        else:
            legend_patch = mpatches.Patch(
                color=colors[shift + 1],
                label=f"{feat_name_display_dict[f]}$_{shift_label}$",
            )
        legend_patches.append(legend_patch)

        # right plot
        ### ax1
        t0_idx = int(1.5 * 50)

        ttrain0 = m[:, t0_idx]
        sig_t0 = sig[:, t0_idx]

        if sig_t0.sum() > 0:
            ax[1].plot(ttrain0, times, color=colors[shift])
            ax[1].fill_betweenx(
                times, ttrain0, where=sig_t0, color=colors[shift], alpha=0.4
            )
            if DEVICE == "Meg":
                ax[1].scatter(
                    np.full(sig_t0.sum(), 0.075 + shift * 0.006),
                    times[sig_t0],
                    color=colors[shift],
                    s=1,
                )
            else:
                ax[1].scatter(
                    np.full(sig_t0.sum(), 0.02 + shift * 0.006),
                    times[sig_t0],
                    color=colors[shift],
                    s=1,
                )
            ax[1].set_ylim(-1, 1)

    # legend outside axis, remove boarder
    axd["seq_l"].legend(
        handles=legend_patches, loc="upper left", bbox_to_anchor=(1.1, 1), frameon=False
    )

    # formatting
    sns.despine(left=True, bottom=True)
    ax[0].set_xlabel("Test Time (s)")
    ax[0].set_ylabel("Train Time (s)")

    # add vertical and horizontal line at t0
    ymin, ymax = ax[0].get_ylim()
    ax[0].vlines(0, ymin, ymax, colors="grey", linewidth=0.5)
    xmin, xmax = ax[0].get_xlim()
    ax[0].hlines(0, xmin, xmax, colors="grey", linewidth=0.5)

    ax[0].yaxis.set_major_formatter(FuncFormatter(custom_formatter))
    ax[0].xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    ax[1].xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    ax[1].yaxis.set_major_formatter(FuncFormatter(custom_formatter))


def plot_key01(f, score, ax, positions=(0, 1)):
    """Standalone contour panel showing only the requested key positions
    (e.g. Key0 and Key1), using the same procedure as the gen panel."""
    # crop to the same time window as the gen panel before stats / FDR
    times = np.linspace(-1.5, 1.5, score.shape[-1])
    sel = np.logical_and(times > -1, times < 1.5)
    score = score[..., sel][..., sel, :]

    means, pvals = Stats(feature=f).run(score)
    n_shifts = len(means)
    center = n_shifts // 2

    times = np.linspace(-1, 1.5, means.shape[-1])

    shifts = np.flip(np.arange(n_shifts))

    for shift in shifts:
        # key position label matches plot_rainbow (shift_label = center - shift)
        shift_label = center - shift
        if shift_label not in positions:
            continue

        p = pvals[shift]
        _, p = fdr_correction(p)

        sig = p < 0.05
        if sig.sum() == 0:
            print(f"{f} key{shift_label} not showing fold")
            continue

        # fill: same method as the seq panel (binary sig, flat light fill)
        fill_cmap = create_custom_cmap(colors[shift + 1], transparencies=[0.0, 0.3])
        ax.pcolormesh(times + (4 - shift) * move, times, sig, cmap=fill_cmap)
        # contour same as the gen panel
        contour_cmap = create_custom_cmap(
            colors[shift + 1], transparencies=[0.0, 0.5, 0.8, 0.9, 1.0]
        )
        ax.contour(
            times + (4 - shift) * move,
            times,
            p,
            cmap=contour_cmap,
            levels=[1e-8, plotting_p],
        )

        # contour centroid (point the label should reference)
        ys, xs = np.where(sig)
        x_pos = (times[xs] + (4 - shift) * move).mean()
        y_pos = times[ys].mean()
        if shift_label < 0:
            label = f"{feat_name_display_dict[f]}$_\\minus$$_{-shift_label}$"
        else:
            label = f"{feat_name_display_dict[f]}$_{shift_label}$"

        # Key0 -> upper-left quadrant, Key1 -> lower-right quadrant (axes fraction)
        label_pos = {0: (0.22, 0.78), 1: (0.78, 0.22)}
        ax.annotate(
            label,
            xy=(x_pos, y_pos),
            xycoords="data",
            xytext=label_pos.get(shift_label, (0.5, 0.5)),
            textcoords="axes fraction",
            color=colors[shift + 1],
            fontsize=11,
            weight="bold",
            ha="center",
            va="center",
            arrowprops=dict(
                arrowstyle="-",
                color=colors[shift + 1],
                lw=0.8,
                shrinkA=5,
                shrinkB=5,
            ),
        )

    # formatting (mirror plot_rainbow's left panel)
    sns.despine(left=True, bottom=True)
    ax.set_xlabel("Test Time (s)")
    ax.set_ylabel("Train Time (s)")

    ymin, ymax = ax.get_ylim()
    ax.vlines(0, ymin, ymax, colors="grey", linewidth=0.5)
    xmin, xmax = ax.get_xlim()
    ax.hlines(0, xmin, xmax, colors="grey", linewidth=0.5)

    ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
    ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))


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

    # annotate contour
    ax.annotate(f"p = {plotting_p}", xy=(-0.5, 0.5), fontsize=10, color=color)

    # remove all spines
    sns.despine(left=True, bottom=True)
    ax.set_ylabel("Train Time (s)")
    ax.set_xlabel("Test Time (s)")

    # plot vertical and horizontal lines at t0
    ymin, ymax = ax.get_ylim()
    ax.vlines(0, ymin, ymax, colors="grey", linewidth=0.5)
    ax.hlines(0, ymin, ymax, colors="grey", linewidth=0.5)
    ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))


def get_decoder_scores(ts: list, score: np.array, t: np.array) -> tuple:

    out = []
    for ti in ts:
        idx = np.argmin(np.abs(t - ti))

        decoder = score[:, :, :, idx].mean((0, 2)).squeeze()
        out.append(decoder)

    return out


def plot_curves(decoder_list, decoder_t, ax_complex, t):
    n_decoders = len(decoder_list)
    n_shifts = decoder_list[0].shape[0]
    # Create the subplots within the 'B' axis

    gs = ax_complex.get_gridspec()
    subaxes = gs[1, 0].subgridspec(n_decoders, 1, hspace=-0.25)

    min_max_scores = [(d.min(), d.max()) for d in decoder_list]
    min_max_scores = np.array(min_max_scores)
    min_score = min_max_scores[:, 0].min()
    max_score = min_max_scores[:, 1].max()

    if DEVICE == "Meg":
        labels = [
            f"Decoder $_{{{int(decoder_t[d]*1e3)}\\,\\mathrm{{ms}}}}$"
            for d in range(n_decoders)
        ]
    else:  # EEG: two-line labels, raised clear of the broader, lower EEG curves
        labels = [
            f"Decoder\n{int(decoder_t[d]*1e3)} ms".replace("-", "\u2212")
            for d in range(n_decoders)
        ]

    for d in range(n_decoders):
        ax_idx = d
        ax = fig.add_subplot(subaxes[ax_idx])
        for s in range(n_shifts):

            ax.plot(t, decoder_list[d][s], color=colors[s])

        ax.set_facecolor((0, 0, 0, 0))

        ax.set_ylim(min_score, max_score)
        if ax_idx != n_decoders - 1:
            ax.xaxis.set_visible(False)
            sns.despine(bottom=True, left=True, ax=ax)
            # remove yticks and yticklabels
            ax.set_yticks([])
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Score")
            ax.set_xlabel("Test Time (s)")
            ax.yaxis.set_visible(True)
            # show y axis spine
            ax.tick_params(axis="y", length=0)
            sns.despine(bottom=True, ax=ax)

        if DEVICE == "Meg":
            ax.annotate(text=labels[d], xy=(-1.4, 0.05), color="black", alpha=1)
        else:  # left of the curves (at baseline before ~-0.7 s), below the panel above
            ax.annotate(
                text=labels[d],
                xy=(t[0], 0.6),
                xycoords=("data", "axes fraction"),
                ha="left",
                va="bottom",
                fontsize=9,
                color="black",
            )

        # set formatter
        ax.yaxis.set_major_formatter(FuncFormatter(custom_formatter))
        ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))


def plot_3d(
    data,
    decoders=[0.0, 0.1, 0.2],
    windows=[-0.3, 0.1, 0.5],
    colors=colors,
    ax_complex=None,
    labelpad=-5,
):
    # mean across subjects
    mean = data.mean(dim=["split", "subject"])
    # divide by max score
    mean = (mean - mean.min(dim="test_time")) / (
        mean.max(dim="test_time") - mean.min(dim="test_time")
    )

    gs = ax_complex.get_gridspec()
    subaxes = gs[1, 1:].subgridspec(1, len(windows), wspace=0.1)

    # most prominent key positions to label per subpanel (window order)
    label_keys = {
        0: {-3, -2, -1},
        1: {0, 1},
        2: {2, 3, 4},
    }

    for w in range(len(windows)):
        window = windows[w]
        ax = fig.add_subplot(subaxes[0, w], projection="3d")

        # select data
        xyz = np.asarray(
            [
                mean.
                # select decoder
                sel(train_time=decoder, method="nearest").
                # select time window
                where(
                    (mean.test_time < window) & (mean.test_time > window - 1.0), drop=True
                ).
                # convert to numpy
                to_numpy()
                for decoder in decoders
            ]
        )

        # plot each shift
        for shift, color in enumerate(colors[1:-1]):
            plot_fading(ax, xyz[:, shift, :], color=color)

            # label the most prominent keys for this subpanel at the final marker
            key_label = (len(colors[1:-1]) // 2) - shift  # matches legend convention
            if key_label in label_keys.get(w, set()):
                x_end, y_end, z_end = xyz[:, shift, -1]
                if key_label < 0:
                    key_text = (
                        f"{feat_name_display_dict[feat_name]}$_\\minus$$_{-key_label}$"
                    )
                else:
                    key_text = f"{feat_name_display_dict[feat_name]}$_{key_label}$"
                # offset to the upper-left of the dot so labels don't touch it
                off = 0.05
                ax.text(
                    x_end - off,
                    y_end,
                    z_end + off,
                    key_text,
                    color=color,
                    fontsize=11,
                    weight="bold",
                    ha="right",
                    va="bottom",
                    zorder=10,
                )

        # clean up
        ax.set_title(f"{int(window*1e3)} ms", fontsize=12)
        if w == 1:
            ax.set_title(f"{int(window*1e3)} ms (Test Time)", fontsize=12)

        ax.set_xticks([])
        ax.set_yticks([])
        if ax.name == "3d":
            ax.set_zticks([])

        ax.set_xlabel("D$_{{-100}}$", labelpad=labelpad)
        ax.set_ylabel("D$_{{0}}$", labelpad=labelpad)
        ax.set_zlabel("D$_{{100}}$", labelpad=labelpad)


# %%
gs_kw = dict(width_ratios=[1.3, 1.3, 2.0, 0.6], hspace=0.15, wspace=0.15, height_ratios=[0.7, 1])
fig, axd = plt.subplot_mosaic(
    [["gen", "key01", "seq", "seq_l"], ["curve", "rota", "rota", "rota"]],
    gridspec_kw=gs_kw,
    figsize=(12, 5.3333333),
    layout="constrained",
)

plot_rainbow(feat_name, sequential_score, [axd["seq"], axd["seq_l"]])

plot_timegen(feat_name, timegen_score, axd["gen"])
axd["gen"].set_xlim(-0.75, 1.0)
axd["gen"].set_ylim(-0.75, 1.0)

# intermediate panel: standalone Key0 and Key1 pulled from the rainbow overlay
plot_key01(feat_name, sequential_score, axd["key01"], positions=(0, 1))
# zoom to exactly match the gen panel
axd["key01"].set_xlim(axd["gen"].get_xlim())
axd["key01"].set_ylim(axd["gen"].get_ylim())

axd["seq_l"].set_xlabel("Score")
axd["seq_l"].set_ylabel("Train Time (s)")


# plot curve plot
t = np.arange(exp.start, exp.start + exp.duration, 1 / exp.device.frequency)

decoder_t = [-0.1, 0.0, 0.1]
decoder_list = get_decoder_scores(decoder_t, sequential_score, t)
plot_curves(decoder_list, decoder_t, axd["curve"], t)
axd["curve"].remove()


# choose ts
ts = [
    60,
    80,
    100,
]

rotation_data = get_data(sequential_score)
plot_3d(rotation_data, decoders=decoder_t, ax_complex=axd["rota"], labelpad=-15)

axd["rota"].remove()

# add color bar
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# add colorbar for seq
grey_cmap = colorcet.cm.gray_r
norm = mpl.colors.Normalize(vmin=0, vmax=1)


# add colorbar for gen
color = feat_color[feat_name]
gradient_cmap = create_custom_cmap(color, transparencies=[0.0, 0.5, 0.8, 0.9, 1.0])

gen_score_mean = timegen_score.mean((0, 1))
norm = mpl.colors.Normalize(vmin=gen_score_mean.min(), vmax=gen_score_mean.max())
cax = inset_axes(
    axd["gen"],
    width="3%",
    height="20%",
    loc="lower right",
    bbox_to_anchor=(0.0, 0.0, 0.92, 1.0),
    bbox_transform=axd["gen"].transAxes,
)

fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=gradient_cmap),
    cax=cax,
    orientation="vertical",
    label="Score",
)


label_fontsize = 12


plt.savefig(fig_path / f"fig3-{DEVICE}-v2.png", bbox_inches="tight")
plt.show()


# %%
