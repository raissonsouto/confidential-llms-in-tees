"""Azure counterpart to price.py.

price.py computes vCPU-per-dollar for GCP c4-highcpu sizes from hardcoded
prices. This does the same for the two Azure VM families this project
actually uses (see ../../AZURE.md and ../../.env):

  - DCsv3   (Standard_DC*s_v3)  -- SGX VM / native baseline, region eastus2
  - DCesv6  (Standard_DC*es_v6) -- TDX VM,                   region westus3

Prices are fetched live from Azure's public Retail Prices API (no auth
required) instead of being hardcoded, since Azure pricing changes over time
and baking in a snapshot would silently go stale. Only Linux, pay-as-you-go
("Consumption"), non-Spot prices are used -- matching how the project's own
VMs were actually provisioned (see AZURE.md).

Usage: python3 price_azure.py
Writes ../../results/azure_price.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import sys
import urllib.parse
import urllib.request
import json

API = "https://prices.azure.com/api/retail/prices"

# (family label, armRegionName, productName filter, our actual benchmark SKU)
FAMILIES = [
    ("DCsv3 (SGX VM / baseline)", "eastus2", "DCsv3 Series Linux", "Standard_DC16s_v3"),
    ("DCesv6 (TDX VM)", "westus3", "Virtual Machines DCesv6 Series", "Standard_DC16es_v6"),
]

SIZE_RE = re.compile(r"Standard_DC(\d+)e?s_v\d+")


def fetch_family(region, product_name):
    filt = (f"armRegionName eq '{region}' and serviceName eq 'Virtual Machines' "
            f"and productName eq '{product_name}' and priceType eq 'Consumption'")
    url = f"{API}?$filter={urllib.parse.quote(filt)}"
    prices = {}
    while url:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.load(resp)
        for item in data["Items"]:
            name = item["meterName"]
            if "Spot" in name or "Low Priority" in name:
                continue
            m = SIZE_RE.match(item["armSkuName"])
            if not m:
                continue
            vcpus = int(m.group(1))
            prices[vcpus] = item["retailPrice"]
        url = data.get("NextPageLink")
    return prices


def main():
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(11, 4.5))
    colors = ['#2a78d6', '#eda100']

    for ax, (label, region, product_name, our_sku), color in zip(axes, FAMILIES, colors):
        try:
            prices = fetch_family(region, product_name)
        except Exception as exc:
            print(f"could not fetch pricing for {label} ({region}): {exc}", file=sys.stderr)
            ax.text(0.5, 0.5, "pricing API\nunreachable", ha='center', va='center',
                    transform=ax.transAxes, color='#898781')
            ax.set_title(label)
            continue

        if not prices:
            print(f"no prices returned for {label} ({region})", file=sys.stderr)
            continue

        vcpus = sorted(prices)
        vcpu_per_dollar = [v / prices[v] for v in vcpus]

        print(f"\n{label} ({region}), Linux on-demand:")
        for v in vcpus:
            print(f"  {v:>4} vCPU: ${prices[v]:.4f}/hr  ({v / prices[v]:.2f} vCPU/US$)")

        bars = ax.bar([str(v) for v in vcpus], vcpu_per_dollar, color=color)
        for bar, v in zip(bars, vcpus):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.1f}',
                    ha='center', va='bottom', fontsize=8)
        ax.set_title(label)
        ax.set_xlabel("vCPUs")
        ax.set_ylabel("vCPU per US$ per hour")
        ax.grid(axis='y')
        ax.set_axisbelow(True)

        # This project's experiment ran at 16 vCPUs on exactly `our_sku`.
        our_vcpus = int(SIZE_RE.match(our_sku).group(1))
        if our_vcpus in prices:
            print(f"  -> this experiment's VM ({our_sku}): ${prices[our_vcpus]:.4f}/hr "
                  f"for {our_vcpus} vCPUs")

    plt.tight_layout()
    plt.savefig("../../results/azure_price.png", bbox_inches='tight', transparent=True, pad_inches=0, dpi=150)
    print("\nwrote ../../results/azure_price.png")


if __name__ == "__main__":
    main()
