#!/usr/bin/env python3
import argparse
import re
import sys
import glob
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # 캡션이나 추가 플롯을 위해 남겨둡니다.
from matplotlib.colors import LinearSegmentedColormap

# Datashader 및 관련 모듈 임포트 (멀티프로세싱을 위해 Dask 스케줄러를 'processes'로 설정)
try:
    import datashader.transfer_functions as tf
    import colorcet as cc
    import dask
    import dask.array as da
    # 멀티프로세싱을 위해 scheduler를 processes로 설정
    dask.config.set(scheduler='processes')
except ImportError as e:
    print("ImportError:", e)
    sys.exit(1)

def parse_node_pfn_stats(filepath='/sys/kernel/debug/numa_folio/pfn_stats'):
    """
    /sys/kernel/debug/numa_folio/pfn_stats 파일을 파싱하여 각 NUMA 노드의 PFN 범위를 반환한다.
    예: {0: (start_pfn, end_pfn), 1: (start_pfn, end_pfn), ...}
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
            m = re.search(r'start pfn:?\s*(\d+)[,]?\s*(?:end pfn:?\s*(\d+))', line)
            if m and (current_node is not None):
                node_ranges[current_node] = (int(m.group(1)), int(m.group(2)))
                current_node = None
    return node_ranges

def combine_logs_and_generate_heatmaps(log_pattern="folio_logs/folio_stats_snapshot_*.log",
                                         node_pfn_stats_path="/sys/kernel/debug/numa_folio/pfn_stats"):
    """
    로그 파일을 결합하여 DataFrame을 생성하고,
    각 NUMA 노드별로 전체 PFN 범위를 반영한 pivot 테이블을 만든 후,
    제일 마지막 스냅샷의 migrate_count 최댓값(정수)을 기준으로 Datashader를 이용하여
    멀티프로세싱으로 히트맵을 생성하고 PNG 파일로 저장한다.
    
    또한, 전체 데이터를 CSV 파일로 저장한다.
    """
    # 스냅샷 번호 순으로 로그 파일 정렬
    log_files = sorted(glob.glob(log_pattern),
                       key=lambda x: int(re.search(r"folio_stats_snapshot_(\d+)\.log", x).group(1)))
    if not log_files:
        print("No log files found matching pattern", log_pattern)
        sys.exit(1)

    collected_data = []
    # folio 통계 라인을 추출하기 위한 정규식
    line_regex = re.compile(r"folio node:?\s*(\d+),\s*pfn:?\s*(\d+),\s*source_nid:?\s*(\d+),\s*migrate_count:?\s*(\d+)")
    
    # 각 로그 파일에서 데이터 추출
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

    # DataFrame 생성 및 CSV 저장
    df = pd.DataFrame(collected_data, columns=["node", "pfn", "source_nid", "migrate_count", "snapshot"])
    
    # PFN 범위 정보 파싱
    node_ranges = parse_node_pfn_stats(node_pfn_stats_path)
    print("Node PFN ranges:")
    print(node_ranges)
    
    # 총 스냅샷 수 확인
    if df.empty:
        all_snapshots = range(0, 2)
    else:
        all_snapshots = range(df["snapshot"].min(), df["snapshot"].max() + 1)
    print("Total number of snapshots collected:", len(all_snapshots))
    
    # 마지막 스냅샷의 migrate_count 최댓값을 global_vmax로 설정 (정수)
    last_snapshot = df["snapshot"].max()
    global_vmax = int(df[df["snapshot"] == last_snapshot]["migrate_count"].max())
    
    # 각 노드별 히트맵 생성
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
            # 전체 PFN 범위를 재설정
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

        # 데이터 배열을 dask.array로 변환하여 병렬 처리 활성화
        # 여기서 chunk 크기는 데이터 크기에 따라 "auto"로 설정하거나, 직접 지정할 수 있음
        darr = da.from_array(pivot.to_numpy(), chunks="auto")

        # Datashader의 tf.shade()를 이용하여 히트맵 이미지 렌더링
        img = tf.shade(darr, cmap=cc.fire, how='linear')

        final_filename = f"node_{node}_migration_heatmap.png"
        # PIL 이미지 객체로 변환 후 저장
        img.to_pil().save(final_filename)
        print(f"Final heatmap for node {node} saved as '{final_filename}'.")

    return df

def main():
    parser = argparse.ArgumentParser(
        description="Combine folio_stats_snapshot log files to build a DataFrame and generate migration heatmaps using Datashader with multiprocessing."
    )
    parser.add_argument("--log_pattern", type=str, default="folio_logs/folio_stats_snapshot_*.log",
                        help="Glob pattern for log files (default: folio_logs/folio_stats_snapshot_*.log)")
    parser.add_argument("--node_pfn_stats", type=str, default="/sys/kernel/debug/numa_folio/pfn_stats",
                        help="Path to the node_pfn_stats file (default: /sys/kernel/debug/numa_folio/pfn_stats)")
    args = parser.parse_args()

    df = combine_logs_and_generate_heatmaps(args.log_pattern, args.node_pfn_stats)
    # 결합된 DataFrame을 CSV 파일로 저장
    df.to_csv("combined_folio_stats.csv", index=False)
    print("Combined data saved as 'combined_folio_stats.csv'.")

if __name__ == "__main__":
    main()
