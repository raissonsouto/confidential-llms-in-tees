#!/usr/bin/env python3
"""Summarise a GPU sweep: latency, throughput and cost per million tokens.

Reads the latency_in<IN>_bs<BS>.json files that benchmark_vllm.sh produces and
prints one table per result directory, plus a confidential-vs-baseline overhead
column when both arms are given.

    python3 parse.py ../results/gpu ../results/cgpu
"""

import json
import glob
import re
import sys
import os

# a3-highgpu-1g (1x H100 80GB, 26 vCPU, 234 GB), Spot, us-central1.
#
# Confidential VM with an H100 is only offered under the Spot and flex-start
# provisioning models, so the spot rate -- not the on-demand rate -- is the one
# that applies to both arms here. Upstream used $6.98/h, the Azure H100 NVL
# on-demand rate, which does not apply on GCP.
#
# Spot prices move; override with GPU_COST_PER_HOUR to price a specific run, and
# check the current rate at https://cloud.google.com/compute/gpus-pricing
COST = float(os.environ.get("GPU_COST_PER_HOUR", "10.094"))

OUTPUT_LEN = 128
PATTERN = re.compile(r"latency_in(\d+)_bs(\d+)\.json$")


def load(results_dir):
    """Return {(input_len, batch_size): avg_latency} for one result directory."""
    out = {}
    for path in glob.glob(os.path.join(results_dir, "latency_in*_bs*.json")):
        match = PATTERN.search(os.path.basename(path))
        if not match:
            continue
        with open(path, "r") as f:
            data = json.load(f)
        avg_latency = data.get("avg_latency")
        if avg_latency is None:
            continue
        out[(int(match.group(1)), int(match.group(2)))] = avg_latency
    return out


def report(label, latencies):
    print(f"\n{label}  (${COST:.3f}/hour)")
    print(f"  {'input':>6} {'batch':>6} {'latency(s)':>11} {'tok/s':>10} {'$/Mtok':>10}")
    for key in sorted(latencies):
        input_len, batch_size = key
        avg_latency = latencies[key]
        # Throughput is generated tokens per second: the benchmark times the
        # generation of OUTPUT_LEN tokens for each of batch_size sequences.
        tokens_per_sec = OUTPUT_LEN * batch_size / avg_latency
        cost_per_million = COST / (tokens_per_sec * 3600) * 1e6
        print(f"  {input_len:>6} {batch_size:>6} {avg_latency:>11.3f} "
              f"{tokens_per_sec:>10.1f} {cost_per_million:>10.3f}")


def overhead(baseline, confidential):
    shared = sorted(set(baseline) & set(confidential))
    if not shared:
        return
    print("\nConfidential GPU overhead (throughput loss vs baseline)")
    print(f"  {'input':>6} {'batch':>6} {'gpu tok/s':>11} {'cgpu tok/s':>11} {'overhead':>10}")
    for key in shared:
        input_len, batch_size = key
        gpu_tps = OUTPUT_LEN * batch_size / baseline[key]
        cgpu_tps = OUTPUT_LEN * batch_size / confidential[key]
        print(f"  {input_len:>6} {batch_size:>6} {gpu_tps:>11.1f} {cgpu_tps:>11.1f} "
              f"{cgpu_tps / gpu_tps - 1:>9.2%}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    loaded = []
    for results_dir in sys.argv[1:]:
        latencies = load(results_dir)
        if not latencies:
            print(f"warning: no latency_in*_bs*.json with an avg_latency in {results_dir}",
                  file=sys.stderr)
        report(os.path.basename(os.path.normpath(results_dir)), latencies)
        loaded.append((results_dir, latencies))

    # If exactly two directories were given, treat the one whose name mentions
    # "cgpu" as the confidential arm and compare.
    if len(loaded) == 2:
        (dir_a, lat_a), (dir_b, lat_b) = loaded
        if "cgpu" in os.path.basename(os.path.normpath(dir_b)):
            overhead(lat_a, lat_b)
        elif "cgpu" in os.path.basename(os.path.normpath(dir_a)):
            overhead(lat_b, lat_a)


if __name__ == "__main__":
    main()
