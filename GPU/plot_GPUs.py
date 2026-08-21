#!/usr/bin/env python3
"""Plot baseline-GPU vs confidential-GPU throughput for the reduced sweep.

    python3 plot_GPUs.py ../results/gpu ../results/cgpu

Left panel:  throughput vs batch size at a fixed input length.
Right panel: throughput vs input length at a fixed batch size.
Confidential bars are annotated with their overhead relative to the baseline.
"""
import os
import re
import sys
import glob
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ─── Configuration ──────────────────────────────────────────────────────────────
# The reduced sweep is batch 1/64 x input 128/512/2048, matching CPU/run.sh.
ORDER_B      = ["1", "64"]
FIXED_INPUT  = 128      # input length for the left plot
FIXED_BATCH  = 64       # batch size for the right plot
OUTPUT_LEN   = 128      # generated tokens per sequence, per iteration

HUE_ORDER    = ["GPU", "cGPU"]         # first GPU then cGPU
COLORS       = ['#E07A5F', '#F4A261']  # match GPU, cGPU respectively
PATTERN      = re.compile(r"latency_in(\d+)_bs(\d+)\.json$")
OUT_PNG      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "results", "gpu_throughput_comparison.png")
plt.rcParams.update({'font.size': 12})
# ────────────────────────────────────────────────────────────────────────────────

VM_DIR = sys.argv[1] if len(sys.argv) > 1 else "../results/gpu"
CC_DIR = sys.argv[2] if len(sys.argv) > 2 else "../results/cgpu"

# 1) Load JSONs into a DataFrame
rows = []
for system, d in [("cGPU", CC_DIR), ("GPU", VM_DIR)]:
    for path in glob.glob(os.path.join(d, "latency_in*_bs*.json")):
        m = PATTERN.search(os.path.basename(path))
        if not m:
            continue
        inp = int(m.group(1))
        bs  = int(m.group(2))

        with open(path) as f:
            data = json.load(f)

        latencies = data.get("latencies", [])
        # skip files with no per-sample latencies
        if not latencies:
            continue

        # append one row per-sample, so the bars carry the spread across the
        # 30 measured iterations rather than a single mean
        for lat in latencies:
            rows.append({
                "system":     system,
                "input_len":  inp,
                "batch_size": bs,
                "latency_ms": lat,
                "throughput": OUTPUT_LEN * bs / lat,  # generated tokens per second
            })

df = pd.DataFrame(rows)
if df.empty:
    sys.exit(f"no usable results found in {VM_DIR} or {CC_DIR}")


def annotate_overhead(ax, order):
    """Label each cGPU bar with its throughput delta against the GPU bar."""
    for idx, bar in enumerate(ax.patches):
        if idx // len(order) != 1:      # cGPU bars come after all GPU bars
            continue
        cc_h = bar.get_height()
        vm_h = ax.patches[idx % len(order)].get_height()
        if not vm_h:
            continue
        ov = cc_h / vm_h - 1
        x = bar.get_x() + bar.get_width() / 2
        # Batch 1 and batch 64 differ by more than an order of magnitude, so a
        # label centred inside a short bar spills past the axis. Put it above
        # the bar instead whenever the bar is a small fraction of the panel.
        top = ax.get_ylim()[1]
        if cc_h < 0.25 * top:
            ax.annotate(f"{ov:.2%}", xy=(x, cc_h + 0.02 * top),
                        ha='center', va='bottom', rotation='vertical')
        else:
            ax.annotate(f"{ov:.2%}", xy=(x, cc_h * 0.5),
                        ha='center', va='center', rotation='vertical')


# 2) Prepare figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

# ── Left: throughput vs batch size (fixed input) ────────────────────────────────
df_b = df[df["input_len"] == FIXED_INPUT].copy()
df_b["batch_size"] = df_b["batch_size"].astype(str)

sns.barplot(
    data=df_b, x="batch_size", y="throughput", hue="system",
    hue_order=HUE_ORDER, palette=COLORS, order=ORDER_B, ax=ax1, zorder=2,
)
for idx in range(len(ORDER_B)):
    ax1.axvline(x=idx, color='gray', linestyle='--', zorder=0, alpha=0.5)
annotate_overhead(ax1, ORDER_B)

ax1.set_title(f"input length={FIXED_INPUT}")
ax1.set_xlabel("Batch size")
ax1.set_ylabel("Throughput (tokens/sec)")
ax1.grid(axis='y')
ax1.set_axisbelow(True)
ax1.legend(title="")

# ── Right: throughput vs input length (fixed batch) ────────────────────────────
df_i = df[df["batch_size"] == FIXED_BATCH].copy()
ORDER_I = [str(i) for i in sorted(df_i["input_len"].unique(), key=int)]
df_i["input_len"] = df_i["input_len"].astype(str)

sns.barplot(
    data=df_i, x="input_len", y="throughput", hue="system",
    hue_order=HUE_ORDER, palette=COLORS, order=ORDER_I, ax=ax2, zorder=2,
)
for idx in range(len(ORDER_I)):
    ax2.axvline(x=idx, color='gray', linestyle='--', zorder=0, alpha=0.5)
annotate_overhead(ax2, ORDER_I)

ax2.set_title(f"batch size={FIXED_BATCH}")
ax2.set_xlabel("Input length (tokens)")
ax2.set_ylabel("")
ax2.grid(axis='y')
ax2.set_axisbelow(True)
ax2.legend().remove()

# ── Final touches ───────────────────────────────────────────────────────────────
plt.tight_layout()
plt.savefig(OUT_PNG, bbox_inches='tight', dpi=150)
print(f"wrote {os.path.normpath(OUT_PNG)}")
