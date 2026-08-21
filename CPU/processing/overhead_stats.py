"""
Statistical analysis of the single-socket 7B/bf16 sweep (results.csv): per-configuration
median throughput/latency with bootstrap 95% confidence intervals, and Mann-Whitney U
tests of each TEE against the VM baseline at the same (batch, input length) cell.

Throughput follows the same definition as model_scaling_single_socket.py:
throughput_i = batch_size * out_size / time_i, with the warmup repetition (index 0)
dropped and |z-score| < 3 outlier filtering per system/cell.

Latency reconstructs a per-repetition next-token latency by removing the (per-cell,
not per-repetition) prefill cost reported by the benchmark itself ("First token average
latency" in the raw log) from the total wall time, then dividing by the remaining
decode steps: latency_i = (time_i*1000 - first_token_ms) / (out_size - 1). This
reproduces the medians already reported in Table 4 (verified against the committed
values) and, unlike those single numbers, keeps the per-repetition distribution needed
for a confidence interval.

Usage: python3 overhead_stats.py <results.csv> <results_dir>
  results_dir is the directory containing baseline/, sgx/, tdx/ raw log subfolders.
"""
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(0)
N_BOOT = 10000
ORDER = ["baseline", "sgx", "tdx"]
CELLS = [("1bs", 128), ("1bs", 512), ("1bs", 2048), ("64bs", 128), ("64bs", 512), ("64bs", 2048)]


def first_token_ms(results_dir, system, bs, in_size):
    bs_n = bs[:-2]
    path = f"{results_dir}/{system}/{system}-{in_size}in-128out-16vCPU-1s-{bs}-7b-bf16.txt"
    try:
        text = open(path).read()
    except FileNotFoundError:
        return None
    m = re.search(r"First token average latency:\s*([\d.]+)\s*sec", text)
    return float(m.group(1)) * 1000 if m else None


def load_cell(df, results_dir, system, bs, in_size):
    sub = df[(df.system == system) & (df.bs == bs) & (df.in_size == in_size)]
    sub = sub[sub["index"] != 0]
    if sub.empty:
        return None
    z = np.abs(stats.zscore(sub["time"]))
    sub = sub[z < 3]
    if sub.empty:
        return None
    bs_n = int(bs[:-2])
    out_size = sub["out_size"].iloc[0]
    throughput = (bs_n * out_size / sub["time"]).to_numpy()
    ftl = first_token_ms(results_dir, system, bs, in_size)
    latency = None
    if ftl is not None:
        latency = (sub["time"].to_numpy() * 1000 - ftl) / (out_size - 1)
    return throughput, latency


def bootstrap_median_ci(x, n_boot=N_BOOT):
    boots = RNG.choice(x, size=(n_boot, len(x)), replace=True)
    meds = np.median(boots, axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return np.median(x), lo, hi


def bootstrap_overhead_ci(baseline, tee, n_boot=N_BOOT):
    b_boots = RNG.choice(baseline, size=(n_boot, len(baseline)), replace=True)
    t_boots = RNG.choice(tee, size=(n_boot, len(tee)), replace=True)
    ratios = 1 - np.median(t_boots, axis=1) / np.median(b_boots, axis=1)
    point = 1 - np.median(tee) / np.median(baseline)
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return point, lo, hi


def main():
    results_csv, results_dir = sys.argv[1], sys.argv[2]
    df = pd.read_csv(results_csv)

    cells = {}
    for system in ORDER:
        for bs, in_size in CELLS:
            r = load_cell(df, results_dir, system, bs, in_size)
            if r is not None:
                cells[(system, bs, in_size)] = r

    print(f"{'system':8} {'bs':4} {'in':5} {'thr_med':8} {'thr_CI95':18} "
          f"{'lat_med':9} {'lat_CI95':20} {'overhead%':10} {'ov_CI95':18} {'p_thr':10} {'p_lat':10}")
    for bs, in_size in CELLS:
        if ("baseline", bs, in_size) not in cells:
            continue
        base_thr, base_lat = cells[("baseline", bs, in_size)]
        for system in ["baseline", "sgx", "tdx"]:
            if (system, bs, in_size) not in cells:
                continue
            thr, lat = cells[(system, bs, in_size)]
            thr_med, thr_lo, thr_hi = bootstrap_median_ci(thr)
            lat_str = "n/a"
            p_lat = float("nan")
            if lat is not None:
                lat_med, lat_lo, lat_hi = bootstrap_median_ci(lat)
                lat_str = f"[{lat_lo:.1f},{lat_hi:.1f}]"
                lat_disp = f"{lat_med:.1f}"
            else:
                lat_disp = "n/a"
            if system == "baseline":
                ov_disp, ov_ci, p_thr, p_lat_disp = "---", "---", "---", "---"
            else:
                ov, ov_lo, ov_hi = bootstrap_overhead_ci(base_thr, thr)
                ov_disp = f"{ov*100:.1f}"
                ov_ci = f"[{ov_lo*100:.1f},{ov_hi*100:.1f}]"
                p_thr = stats.mannwhitneyu(base_thr, thr, alternative="two-sided").pvalue
                p_thr = f"{p_thr:.1e}"
                if lat is not None and base_lat is not None:
                    p_lat_v = stats.mannwhitneyu(base_lat, lat, alternative="two-sided").pvalue
                    p_lat_disp = f"{p_lat_v:.1e}"
                else:
                    p_lat_disp = "n/a"
            print(f"{system:8} {bs:4} {in_size:5} {thr_med:<8.2f} {f'[{thr_lo:.2f},{thr_hi:.2f}]':18} "
                  f"{lat_disp:9} {lat_str:20} {ov_disp:10} {ov_ci:18} {p_thr:10} {p_lat_disp:10}")


if __name__ == "__main__":
    main()
