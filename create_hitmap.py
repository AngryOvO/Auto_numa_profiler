#!/usr/bin/env python3
import argparse
import re
import sys
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def parse_node_pfn_stats(filepath='/sys/kernel/debug/numa_folio/pfn_stats'):
    """
    /sys/kernel/debug/numa_folio/pfn_stats 파일을 파싱하여 각 NUMA 노드의 PFN 범위를 반환한다.
    출력 예: {0: (start_pfn, end_pfn), 1: (start_pfn, end_pfn), ...}
    """
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
            m = re.search(r'start pfn\s+(\d+)[,]?\s*(?:end pfn\s+)?(\d+)', line)
            if m and current_node is not None:
                node_ranges[current_node] = (int(m.group(1)), int(m.group(2)))
                current_node = None
    return node_ranges

def combine_logs_and_generate_heatmaps(log_pattern="folio_stats_snapshot_*.log",
                                         node_pfn_stats_path="/sys/kernel/debug/numa_folio/pfn_stats"):
    """
    log_pattern에 맞는 모든 로그 파일을 결합하여 DataFrame을 생성하고,
    각 NUMA 노드별 히트맵을 생성하여 PNG 파일로 저장한다.
    """
    # glob를 사용해 로그 파일 목록을 가져온다.
    log_files = sorted(glob.glob(log_pattern),
                       key=lambda x: int(re.search(r"folio_stats_snapshot_(\d+)\.log", x).group(1)))
    if not log_files:
        print("No log files found matching pattern", log_pattern)
        sys.exit(1)

    collected_data = []
    # 각 로그 파일 내에서 folio 통계 라인을 추출하기 위한 정규식 (공백이나 콜론의 위치가 유연하도록 허용)
    line_regex = re.compile(r"folio node:?\s*(\d+),\s*pfn:?\s*(\d+),\s*source_nid:?\s*(\d+),\s*migrate_count:?\s*(\d+)")
    
    # 각 로그 파일마다 데이터를 추출
    for log_file in log_files:
        # 파일 이름에서 스냅샷 번호 추출
        m_snapshot = re.search(r"folio_stats_snapshot_(\d+)\.log", log_file)
        if not m_snapshot:
            continue
        snapshot = int(m_snapshot.group(1))
        with open(log_file, "r") as f:
            lines = f.readlines()
            for line in lines:
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

    # DataFrame 생성
    df = pd.DataFrame(collected_data, columns=["node", "pfn", "source_nid", "migrate_count", "snapshot"])
    print("Collected data sample:")
    print(df.head())

    # /sys/kernel/debug/numa_folio/pfn_stats 파일을 파싱하여 노드별 PFN 범위 확보
    node_ranges = parse_node_pfn_stats(node_pfn_stats_path)
    print("Node PFN ranges:")
    print(node_ranges)

    # 스냅샷 범위 설정 (최소 ~ 최대 스냅샷)
    if df.empty:
        all_snapshots = range(0, 2)
    else:
        all_snapshots = range(df["snapshot"].min(), df["snapshot"].max() + 1)

    # 모든 노드에 대해 전체 migrate_count의 최대값을 구함 (히트맵 색상 범위에 사용)
    global_vmax = df["migrate_count"].max() if not df.empty else 1
    print(f"Global vmax (maximum migrate_count): {global_vmax}")

    # 노드별로 히트맵 생성
    for node in node_ranges.keys():
        node_df = df[df["node"] == node]
        if not node_df.empty:
            pivot = node_df.pivot_table(
                index="pfn",
                columns="snapshot",
                values="migrate_count",
                aggfunc="sum",
                fill_value=0
            ).fillna(0)
            # 노드의 전체 PFN 범위를 포함하도록 인덱스 강제 재설정
            start_pfn, end_pfn = node_ranges[node]
            full_pfn_range = range(start_pfn, end_pfn + 1)
            pivot = pivot.reindex(index=full_pfn_range, fill_value=0)
        else:
            print(f"No migration data for node {node}. Generating empty heatmap.")
            start_pfn, end_pfn = node_ranges[node]
            full_pfn_range = range(start_pfn, end_pfn + 1)
            pivot = pd.DataFrame(0, index=full_pfn_range, columns=all_snapshots)

        # 히트맵 생성: 열화상 스타일 컬러맵 사용
        cmap = LinearSegmentedColormap.from_list("Thermal", ["navy", "red", "yellow"], N=256)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot,
            cmap=cmap,
            cbar=True,
            vmin=0,
            vmax=global_vmax,
            annot=True,
            fmt="d"
        )
        plt.title(f"Node {node} - Migration Heatmap")
        plt.xlabel("Snapshot (Time)")
        plt.ylabel("PFN")
        plt.tight_layout()
        out_filename = f"node_{node}_migration_heatmap.png"
        plt.savefig(out_filename)
        print(f"Heatmap for node {node} saved as '{out_filename}'.")
        plt.close()

    return df

def main():
    parser = argparse.ArgumentParser(
        description="Combine folio_stats_snapshot log files to build a DataFrame and generate migration heatmaps."
    )
    parser.add_argument("--log_pattern", type=str, default="folio_stats_snapshot_*.log",
                        help="Glob pattern for identifying log files (default: folio_stats_snapshot_*.log)")
    parser.add_argument("--node_pfn_stats", type=str, default="/sys/kernel/debug/numa_folio/pfn_stats",
                        help="Path to the node_pfn_stats file (default: /sys/kernel/debug/numa_folio/pfn_stats)")
    args = parser.parse_args()

    df = combine_logs_and_generate_heatmaps(args.log_pattern, args.node_pfn_stats)
    # 결합된 데이터를 CSV 파일로 저장(선택사항)
    df.to_csv("combined_folio_stats.csv", index=False)
    print("Combined data saved as 'combined_folio_stats.csv'.")

if __name__ == "__main__":
    main()
