#!/usr/bin/env python3
import argparse
import re
import sys
import glob
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    import datashader.transfer_functions as tf
    import colorcet as cc
    import dask.array as da
    from dask.distributed import Client  # 반드시 main() 내부에서 사용할 것
except ImportError as e:
    print("ImportError:", e)
    sys.exit(1)

def parse_node_pfn_stats(filepath='/sys/kernel/debug/numa_folio/pfn_stats'):
    node_ranges = {}
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        sys.exit(1)

    current_node = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("node"):
            parts = line.split()
            try:
                current_node = int(parts[1])
            except ValueError:
                current_node = None
        elif line.startswith("start pfn"):
            m = re.search(r'start pfn:?\s*(\d+)[,]?\s*(?:end pfn:?\s*(\d+))', line)
            if m and (current_node is not None):
                node_ranges[current_node] = (int(m.group(1)), int(m.group(2)))
                current_node = None
    return node_ranges

def combine_logs_and_generate_heatmaps(log_pattern="folio_logs/folio_stats_snapshot_*.log",
                                         node_pfn_stats_path="/sys/kernel/debug/numa_folio/pfn_stats"):
    log_files = sorted(glob.glob(log_pattern),
                       key=lambda x: int(re.search(r"folio_stats_snapshot_(\d+)\.log", x).group(1)))
    if not log_files:
        print("No log files found matching pattern", log_pattern)
        sys.exit(1)

    collected_data = []
    line_regex = re.compile(r"folio node:?\s*(\d+),\s*pfn:?\s*(\d+),\s*source_nid:?\s*(\d+),\s*migrate_count:?\s*(\d+)")
    
    for log_file in log_files:
        m_snapshot = re.search(r"folio_stats_snapshot_(\d+)\.log", log_file)
        if not m_snapshot:
            continue
        snapshot = int(m_snapshot.group(1))
        with open(log_file, "r") as f:
            for line in f:
                m = line_regex.search(line)
                if m:
                    node = int(m.group(1))
                    pfn = int(m.group(2))
                    source_nid = int(m.group(3))
                    migrate_count = int(m.group(4))
                    collected_data.append([node, pfn, source_nid, migrate_count, snapshot])
    
    if not collected_data:
        print("No data extracted from log files.")
        sys.exit(1)

    df = pd.DataFrame(collected_data, columns=["node", "pfn", "source_nid", "migrate_count", "snapshot"])
    node_ranges = parse_node_pfn_stats(node_pfn_stats_path)
    print("Node PFN ranges:")
    print(node_ranges)
    
    if df.empty:
        all_snapshots = range(0, 2)
    else:
        all_snapshots = range(df["snapshot"].min(), df["snapshot"].max() + 1)
    print("Total number of snapshots collected:", len(all_snapshots))
    
    last_snapshot = df["snapshot"].max()
    global_vmax = int(df[df["snapshot"] == last_snapshot]["migrate_count"].max())
    
    for node in node_ranges.keys():
        node_df = df[df["node"] == node]
        if not node_df.empty:
            pivot = node_df.pivot_table(
                index="pfn",
                columns="snapshot",
                values="migrate_count",
                aggfunc="sum",
                fill_value=0
            )
            start_pfn, end_pfn = node_ranges[node]
            full_pfn_range = range(start_pfn, end_pfn + 1)
            pivot = pivot.reindex(index=full_pfn_range, fill_value=0)
            pivot = pivot.astype(int)
        else:
            print(f"No migration data for node {node}. Generating empty heatmap.")
            start_pfn, end_pfn = node_ranges[node]
            full_pfn_range = range(start_pfn, end_pfn + 1)
            pivot = pd.DataFrame(0, index=full_pfn_range, columns=all_snapshots)

        print(f"Node {node} pivot shape: {pivot.shape}")
        darr = da.from_array(pivot.to_numpy(), chunks="auto")
        # Datashader를 통한 병렬 렌더링 – Dask distributed 클라이언트를 사용하여 모든 코어를 활용
        img = tf.shade(darr, cmap=cc.fire, how='linear')
        final_filename = f"node_{node}_migration_heatmap.png"
        img.to_pil().save(final_filename)
        print(f"Final heatmap for node {node} saved as '{final_filename}'.")
    return df

def main():
    # 여기서 Dask Distributed Client를 생성합니다.
    from dask.distributed import Client
    client = Client(n_workers=os.cpu_count(), threads_per_worker=1)
    print("Dask Distributed Client started:")
    print(client)

    parser = argparse.ArgumentParser(
        description="Combine folio_stats_snapshot log files to build a DataFrame and generate migration heatmaps using Datashader with multiprocessing."
    )
    parser.add_argument("--log_pattern", type=str, default="folio_logs/folio_stats_snapshot_*.log",
                        help="Glob pattern for log files (default: folio_logs/folio_stats_snapshot_*.log)")
    parser.add_argument("--node_pfn_stats", type=str, default="/sys/kernel/debug/numa_folio/pfn_stats",
                        help="Path to the node_pfn_stats file (default: /sys/kernel/debug/numa_folio/pfn_stats)")
    args = parser.parse_args()

    df = combine_logs_and_generate_heatmaps(args.log_pattern, args.node_pfn_stats)
    df.to_csv("combined_folio_stats.csv", index=False)
    print("Combined data saved as 'combined_folio_stats.csv'.")

if __name__ == '__main__':
    main()
