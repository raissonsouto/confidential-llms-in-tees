import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Adapted for this reproduction's data. The upstream script swept vCPU count
# (2..192) on the x-axis, faceted by batch size, to find the crossover point
# where CPU cost undercuts a confidential GPU. This run only has one vCPU
# count (16, fixed for the whole experiment), so there is no sweep to plot --
# x-axis becomes "system" instead, still faceted by batch size, at the one
# input length where batch 64 completed for all three systems (input=128;
# SGX/TDX OOM'd at 512/2048 batch 64). The $/vCPU-hr and $/GB-hr constants
# below are the upstream script's own cost model (Azure EMR/SPR CPU
# pricing), kept unchanged; they are not re-derived from this machine.
file = sys.argv[1]
plt.rcParams.update({'font.size': 12})

ORDER = ["baseline", "sgx", "tdx"]
LABELS = {"baseline": "VM (baseline)", "sgx": "SGX", "tdx": "TDX"}
COLORS = ['#2a78d6', '#1baf7a', '#eda100']
NUMA = "1s"
MODEL = "7B"
DTYPE = "bf16"
IN_SIZE = 128
BATCHES = ["1", "64"]
VCPU = 16
MEMORY_GB = 128

# H100 confidential-computing GPU cost ($/million tokens), keyed by
# (input_size, batch_size); copied from the upstream vCPUs_*.py scripts.
GPU_CC_COST = {(128, 1): 13.966463202799583, (128, 64): 0.4383705951320425}


def load():
    df = pd.read_csv(file).rename(columns={"model": "size"})
    df = df.loc[df['index'] != 0]
    df = df.loc[df['system'].isin(ORDER)]
    df = df.loc[df['dt'] == DTYPE]
    df = df.loc[df['size'] == MODEL]
    df = df.loc[df['numa'] == NUMA]
    df = df.loc[df['in_size'] == IN_SIZE]
    df = df.copy()
    df['bs_n'] = df['bs'].str[:-2]
    df = df.loc[df['bs_n'].isin(BATCHES)]
    df['throughput'] = df['bs_n'].astype(int) * df['out_size'] / df['time']
    df['cost_emr'] = 1e6 * (VCPU * 0.01152 + MEMORY_GB * 0.001309) / df['throughput'] / 3600
    return df


df = load()

fig, ax = plt.subplots(nrows=2, ncols=len(BATCHES), figsize=(7, 5.5))
fig.suptitle(f"input = {IN_SIZE}, Llama-2-7B, bf16, {VCPU} vCPUs", y=1.0)

for col, bs in enumerate(BATCHES):
    sub = df.loc[df['bs_n'] == bs]
    if set(sub['system'].unique()) != set(ORDER):
        for row in range(2):
            ax[row][col].text(0.5, 0.5, "OOM\n(<3 systems\ncompleted)", ha='center', va='center',
                               transform=ax[row][col].transAxes, fontsize=10, color='#898781')
            ax[row][col].set_xticks([])
            ax[row][col].set_yticks([])
        ax[0][col].set_title(f"batch = {bs}")
        continue

    sns.barplot(data=sub, x="system", hue="system", y="throughput", order=ORDER, palette=COLORS,
                ax=ax[0][col], legend=False)
    ax[0][col].set_title(f"batch = {bs}")
    ax[0][col].set_xlabel("")
    ax[0][col].set_xticks([])
    ax[0][col].grid(axis='y')
    ax[0][col].set_axisbelow(True)
    ax[0][col].set_ylabel("Throughput\n(tokens/s)" if col == 0 else "")

    sns.barplot(data=sub, x="system", hue="system", y="cost_emr", order=ORDER, palette=COLORS,
                ax=ax[1][col], legend=False)
    key = (IN_SIZE, int(bs))
    if key in GPU_CC_COST:
        ax[1][col].axhline(y=GPU_CC_COST[key], color='#F4A261', linewidth=3,
                            label="confidential H100 (cGPU)")
    ax[1][col].set_xlabel("")
    ax[1][col].set_xticks(range(len(ORDER)))
    ax[1][col].set_xticklabels([LABELS[s] for s in ORDER], rotation=15, ha='right')
    ax[1][col].grid(axis='y')
    ax[1][col].set_axisbelow(True)
    ax[1][col].set_ylabel("Estimated cost\n($/million tokens)" if col == 0 else "")

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
handles.append(plt.Line2D([], [], color='#F4A261', linewidth=3))
fig.legend(handles, [LABELS[s] for s in ORDER] + ["confidential H100 (cGPU)"],
           loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.savefig("../../results/vCPUs_GPU_EMR_batches.png", bbox_inches='tight', transparent=True, pad_inches=0, dpi=150)
print("wrote ../../results/vCPUs_GPU_EMR_batches.png")
