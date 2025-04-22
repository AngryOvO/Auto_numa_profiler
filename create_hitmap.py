import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from collections import defaultdict

# ---------- PFN 범위 읽기 ----------
def load_pfn_ranges_stats(filepath):
    ranges = []
    with open(filepath) as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            node_match = re.search(r'node\s+(\d+)', lines[i])
            range_match = re.search(r'start pfn:\s*(\d+),\s*end pfn:\s*(\d+)', lines[i+1])
            if node_match and range_match:
                node = int(node_match.group(1))
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                ranges.append((node, start, end))
    return sorted(ranges)

# ---------- 로그 파싱 ----------
def parse_logs(log_dir):
    data = defaultdict(list)
    for file in sorted(Path(log_dir).glob('folio_stats_snapshot_*.log')):
        match = re.search(r'(\d+)', file.name)
        if not match:
            continue
        snapshot_idx = int(match.group(1))
        with open(file) as f:
            for line in f:
                tokens = line.strip().split(',')
                if len(tokens) != 3:
                    continue
                pfn = int(tokens[0])
                migrate_count = int(tokens[2])
                data[snapshot_idx].append((pfn, migrate_count))
    return data

# ---------- 히트맵 시각화 ----------
def visualize_heatmap(matrix, node, out_dir='heatmaps'):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, cmap='viridis')
    plt.title(f'Folio Migration Heatmap - Node {node}')
    plt.xlabel('Snapshot Index')
    plt.ylabel('PFN Index (relative to node)')
    plt.colorbar(label='Migrate Count')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/heatmap_node_{node}.png')
    plt.close()

# ---------- 메인 처리 ----------
def main():
    pfn_stats_file = '/sys/kernel/debug/numa_folio/pfn_stats'
    log_dir = 'log_dir_here'  # <- 경로 지정 필요

    ranges = load_pfn_ranges_stats(pfn_stats_file)
    snapshot_data = parse_logs(log_dir)

    max_snap_idx = max(snapshot_data.keys())

    for node, start_pfn, end_pfn in ranges:
        height = end_pfn - start_pfn + 1
        width = max_snap_idx + 1
        matrix = np.zeros((height, width))

        for snap_idx, stats in snapshot_data.items():
            for pfn, count in stats:
                if start_pfn <= pfn <= end_pfn:
                    row = pfn - start_pfn
                    matrix[row, snap_idx] = count

        print(f'Generating heatmap for node {node}...')
        visualize_heatmap(matrix, node)

if __name__ == '__main__':
    main()
