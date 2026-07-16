import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Adapted for this reproduction's data. The upstream version of this script
# imported two helper modules (batch_size_latency.py, batch_size_throughput.py)
# that do not exist in this repo -- and even if they did, this run's
# results.csv only has total per-iteration time (not per-token decode
# latency, see model_scaling_single_socket.py), so a "latency vs batch size"
# panel built the same way as the AMX one would be mislabeled. This script
# instead plots the one metric the data actually supports: throughput vs.
# batch size (1 vs 64), fixed at input=128 -- the only input length where
# batch 64 completed for all three systems (SGX/TDX OOM at 512/2048).
file = sys.argv[1]
plt.rcParams.update({'font.size': 12})

ORDER = ["baseline", "sgx", "tdx"]
LABELS = {"baseline": "VM (baseline)", "sgx": "SGX", "tdx": "TDX"}
COLORS = ['#2a78d6', '#1baf7a', '#eda100']
BS_ORDER = ["1", "64"]
NUMA = "1s"
MODEL = "7B"
DTYPE = "bf16"
IN_SIZE = 128

df = pd.read_csv(file).rename(columns={"model": "size"})
df = df.loc[df['index'] != 0]
df = df.loc[df['system'].isin(ORDER)]
df = df.loc[df['dt'] == DTYPE]
df = df.loc[df['size'] == MODEL]
df = df.loc[df['numa'] == NUMA]
df = df.loc[df['in_size'] == IN_SIZE]
df['bs_n'] = df['bs'].str[:-2]
df = df.loc[df['bs_n'].isin(BS_ORDER)]
df['throughput'] = df['bs_n'].astype(int) * df['out_size'] / df['time']

fig, ax = plt.subplots(figsize=(4.5, 3.5))
sns.barplot(data=df, x="bs_n", hue="system", y="throughput", hue_order=ORDER, order=BS_ORDER,
            palette=COLORS, ax=ax)

for index, p in enumerate(ax.patches):
    height = p.get_height()
    if height > 0 and index >= len(BS_ORDER):
        reference = ax.patches[index % len(BS_ORDER)].get_height()
        ax.annotate(f"{height / reference - 1:.1%}",
                    xy=(p.get_x() + p.get_width() / 2, height + 0.4),
                    ha='center', va='bottom', size=9, rotation='vertical')

handles, _ = ax.get_legend_handles_labels()
ax.legend(handles, [LABELS[s] for s in ORDER], title="")
ax.set_xlabel("Batch size")
ax.set_ylabel("Throughput (tokens/s)")
ax.set_title(f"input = {IN_SIZE}, Llama-2-7B, bf16, 16 vCPUs")
ax.grid(axis='y')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("../../results/batch_scaling_combined.png", bbox_inches='tight', transparent=True, pad_inches=0, dpi=150)
print("wrote ../../results/batch_scaling_combined.png")
