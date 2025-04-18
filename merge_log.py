#!/usr/bin/env python3
import argparse
import re
import sys
import glob
import os
import pandas as pd

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

def combine_logs_and_generate_csv(log_pattern="folio_logs/folio_stats_snapshot_*.log",
                                   node_pfn_stats_path="/sys/kernel/debug/numa_folio/pfn_stats"):
    """
    로그 파일을 결합하여 DataFrame을 생성하고, 
    최종적으로 CSV 파일로 저장한다.
    """
    # 파일명에 포함된 스냅샷 번호 순으로 로그 파일 정렬
    log_files = sorted(glob.glob(log_pattern),
                       key=lambda x: int(re.search(r"folio_stats_snapshot_(\d+)\.log", x).group(1)))
    if not log_files:
        print("No log files found matching pattern", log_pattern)
        sys.exit(1)

    collected_data = []
    # 로그 파일에서 folio 통계 라인을 추출하기 위한 정규식
    line_regex = re.compile(r"folio node:?\s*(\d+),\s*pfn:?\s*(\d+),\s*source_nid:?\s*(\d+),\s*migrate_count:?\s*(\d+)")

    # 각 로그 파일의 데이터를 추출
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

    # DataFrame 생성
    df = pd.DataFrame(collected_data, columns=["node", "pfn", "source_nid", "migrate_count", "snapshot"])

    # 노드별 PFN 범위
    node_ranges = parse_node_pfn_stats(node_pfn_stats_path)
    print("Node PFN ranges:")
    print(node_ranges)
    
    # 총 스냅샷 갯수 출력
    if df.empty:
        all_snapshots = range(0, 2)
    else:
        all_snapshots = range(df["snapshot"].min(), df["snapshot"].max() + 1)
    print("Total number of snapshots collected:", len(all_snapshots))
    
    # DataFrame을 CSV 파일로 저장
    df.to_csv("combined_folio_stats.csv", index=False)
    print("Combined data saved as 'combined_folio_stats.csv'.")

def main():
    parser = argparse.ArgumentParser(
        description="Combine folio_stats_snapshot log files to build a DataFrame and generate CSV output."
    )
    parser.add_argument("--log_pattern", type=str, default="folio_logs/folio_stats_snapshot_*.log",
                        help="Glob pattern for identifying log files (default: folio_logs/folio_stats_snapshot_*.log)")
    parser.add_argument("--node_pfn_stats", type=str, default="/sys/kernel/debug/numa_folio/pfn_stats",
                        help="Path to the node_pfn_stats file (default: /sys/kernel/debug/numa_folio/pfn_stats)")
    args = parser.parse_args()

    combine_logs_and_generate_csv(args.log_pattern, args.node_pfn_stats)

if __name__ == "__main__":
    main()
