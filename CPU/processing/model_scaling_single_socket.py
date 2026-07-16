import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from scipy import stats

# Get the results.csv path from the first argument
# (adapted for this reproduction's reduced sweep: single socket, 16 vCPUs,
#  7B/bf16 only, batch 1bs/64bs, input 128/512/2048 -- no vCPU sweep, no
#  13B/70B, no int8, no dual-socket data available)
file = sys.argv[1]
plt.rcParams.update({'font.size': 12})


def plot_arrows(start, end, height, height_diff, text, ax):
    ax.annotate('', xy=(start, height), xytext=(end, height), arrowprops=dict(arrowstyle="<-", lw=2), zorder=2)
    ax.text((start + end) / 2 - 0.15, height + 0.04 * height_diff, text, ha='center', va='center', fontsize=10,
             backgroundcolor="w", zorder=1, bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='white', lw=0))


def filter_dataframe(batch_size, throughput, data_type, order, model, numa, in_size):
    # Initial load and case filtering. The raw CSV (from run_parser.py) has a
    # "model" column, not "size" -- the upstream plotting scripts filter on
    # "size", so we rename here instead of touching every script.
    df = pd.read_csv(file).rename(columns={"model": "size"})

    df = df.loc[df['index'] != 0]
    df = df.loc[df['system'].isin(order)]
    df = df.loc[df['bs'] == batch_size]
    df = df.loc[df['dt'] == data_type]
    df = df.loc[df['size'] == model]
    df = df.loc[df['numa'] == numa]
    df = df.loc[df['in_size'] == in_size]
    # In this run's results.csv, 'time' is the *total* per-iteration wall
    # time (prefill + all decode steps), not a per-token decode latency --
    # unlike the upstream full sweep, this benchmark always generates the
    # full out_size tokens regardless of batch size. A cell only counts as
    # usable if all three systems actually completed it (OOM cases here
    # drop 2 of 3 systems, not all 3, so checking df.empty alone is not
    # enough).
    if df.empty or set(df['system'].unique()) != set(order):
        return None, None

    bs_n = int(batch_size[:-2])
    out_tok = df['out_size'].iloc[0]
    df['throughput'] = bs_n * out_tok / df['time']

    # Filter outliers (zscore on 'time' per system). Fixed vs. the upstream
    # version: scipy.stats.zscore returns a bare ndarray, so `&`-ing it
    # against the full-length df boolean Series crashes on a shape mismatch
    # once there is more than one system in `order`. Rebuild the mask as an
    # index-aligned Series instead.
    s = 0
    for system in order:
        idx = df.index[df['system'] == system]
        if len(idx) < 2:
            continue
        mask = pd.Series(np.abs(stats.zscore(df.loc[idx, 'time'])) < 3, index=idx)
        drop_idx = mask[~mask].index
        s += len(drop_idx) / len(idx) / len(order)
        df.drop(drop_idx, inplace=True)
    print(f"Filtered: {s * 100:.1f}%")

    # Compute overhead dictionary
    overheads = {system: {system: 0 for system in order} for system in order}
    for system1 in order:
        for system2 in order:
            v1 = df[df["system"] == system1]["throughput"].mean()
            v2 = df[df['system'] == system2]["throughput"].mean()
            overheads[system1][system2] = abs(1 - v1 / v2) if v2 else float("nan")

    return df, overheads


# Define the constants (adapted: our systems are baseline/sgx/tdx, single
# NUMA node "1s", 16 vCPUs, model "7B", dtype "bf16")
ORDER = ["baseline", "sgx", "tdx"]
LABELS = {"baseline": "VM (baseline)", "sgx": "SGX", "tdx": "TDX"}
COLORS = ['#2a78d6', '#1baf7a', '#eda100']
NUMA = "1s"
MODEL = "7B"
DTYPE = "bf16"

# Define variables: columns = input length (the axis of variety we actually
# have), rows = throughput @ batch 64 / throughput @ batch 1. (The upstream
# script's second row was "next-token latency" at batch 1, but this run's
# results.csv only has total per-iteration time, not per-token decode
# latency -- that metric is already covered correctly by
# scripts/fig2_latency.py, which parses the real per-token traces.) 64bs
# only completed at in_size=128 in this run (512/2048 OOM'd on 2 of 3
# systems), so those two cells are left blank and annotated instead of
# computed from partial data.
columns = [128, 512, 2048]
rows = [{"batch_size": "64bs"}, {"batch_size": "1bs"}]

# Define plot
fig, ax = plt.subplots(nrows=len(rows), ncols=len(columns), figsize=(9, 4.5))
plt.subplots_adjust(wspace=0.35, hspace=0.15)
fig.suptitle("Single socket, 16 vCPUs, Llama-2-7B, bf16", y=1.0, x=0.51)

# Plot
for column_index, in_size in enumerate(columns):
    for row_index, row in enumerate(rows):
        df, overheads = filter_dataframe(row["batch_size"], True, DTYPE, ORDER, MODEL, NUMA, in_size)
        cur_ax = ax[row_index][column_index]
        if df is None:
            cur_ax.text(0.5, 0.5, "OOM\n(<3 systems\ncompleted)", ha='center', va='center',
                        transform=cur_ax.transAxes, fontsize=10, color='#898781')
            cur_ax.set_xticks([])
            cur_ax.set_yticks([])
            if row_index != len(rows) - 1:
                cur_ax.set_title(f"input = {in_size}")
            continue

        sns.violinplot(data=df, x="system", hue="system", y="throughput",
                        inner="quart", order=ORDER, palette=COLORS, ax=cur_ax, zorder=2, legend=False)
        for i in range(len(ORDER)):
            cur_ax.axvline(x=i, color='gray', linestyle='--', zorder=0, alpha=0.5)
        cur_ax.set_xlabel("")
        cur_ax.set_ylabel("")
        cur_ax.grid(axis="y")
        cur_ax.set_axisbelow(True)
        cur_ax.set_xticks(range(len(ORDER)))
        cur_ax.set_xticklabels([LABELS[s] for s in ORDER], rotation=15, ha='right')
        if row_index != len(rows) - 1:
            cur_ax.set_xticks([])
            cur_ax.set_title(f"input = {in_size}")

        y_diff = abs(cur_ax.get_ylim()[1] - cur_ax.get_ylim()[0])
        y_min = cur_ax.get_ylim()[0]
        plot_arrows(0, 1, y_min + 0.55 * y_diff, y_diff, f'−{overheads["sgx"]["baseline"]:.1%}', cur_ax)
        plot_arrows(1, 2, y_min + 0.75 * y_diff, y_diff, f'−{overheads["tdx"]["sgx"]:.1%}', cur_ax)
        plot_arrows(0, 2, y_min + 0.92 * y_diff, y_diff, f'−{overheads["tdx"]["baseline"]:.1%}', cur_ax)

ax[0][0].set_ylabel("Throughput\n[tokens/s]\n(batch 64)")
ax[1][0].set_ylabel("Throughput\n[tokens/s]\n(batch 1)")

plt.savefig("../../results/overall_single_socket.png", bbox_inches='tight', transparent=True, pad_inches=0, dpi=150)
print("wrote ../../results/overall_single_socket.png")
