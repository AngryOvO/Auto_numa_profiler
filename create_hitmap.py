#!/usr/bin/env python3
import os
import sys
import re
import argparse
import threading
from dataclasses import dataclass
import numpy as np
import matplotlib

# 헤드리스 환경을 위해 Agg 백엔드 사용
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------- 구조체 정의 (dataclass 사용) ----------------------
@dataclass
class FolioStat:
    pfn: int
    source_pfn: int
    migrate_count: int

@dataclass
class NodeRange:
    node: int
    start: int
    end: int

# ---------------------- 글로벌 변수 ----------------------
# 스냅샷 인덱스 -> 해당 스냅샷의 FolioStat 목록 (딕셔너리)
snapshot_data = {}

# ---------------------- PFN 범위 로딩 (pfn_stats 파일) ----------------------
def load_pfn_ranges_stats(filename, node_ranges):
    try:
        with open(filename, "r") as infile:
            lines = infile.readlines()
    except Exception as e:
        print(f"Error: Unable to open pfn_stats file: {filename}")
        return

    node_pattern = re.compile(r"node\s+(\d+)")
    range_pattern = re.compile(r"start pfn:\s*(\d+),\s*end pfn:\s*(\d+)")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = node_pattern.search(line)
        if m:
            node = int(m.group(1))
            i += 1  # 다음 줄: 범위 정보
            if i < len(lines):
                range_line = lines[i].strip()
                m2 = range_pattern.search(range_line)
                if m2:
                    start = int(m2.group(1))
                    end = int(m2.group(2))
                    node_ranges.append(NodeRange(node=node, start=start, end=end))
                    print(f"Loaded node range: node {node} start: {start} end: {end}")
                else:
                    print(f"Error parsing range line: {range_line}")
        i += 1

# ---------------------- 로그 파일 파싱 ----------------------
def parse_log_file(filepath, snapshot_index):
    stats = []
    try:
        with open(filepath, "r") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                tokens = [token.strip() for token in line.split(",")]
                if len(tokens) == 3:
                    try:
                        # 각 토큰을 정수형으로 변환하여 FolioStat 객체 생성
                        pfn = int(tokens[0])
                        source_pfn = int(tokens[1])
                        migrate_count = int(tokens[2])
                        stats.append(FolioStat(pfn=pfn, source_pfn=source_pfn, migrate_count=migrate_count))
                    except Exception as e:
                        # 파싱 오류 발생 시 해당 라인은 무시
                        pass
        snapshot_data[snapshot_index] = stats
    except Exception as e:
        print(f"Error: Could not open file {filepath}")
        return

# ---------------------- 병렬 로그 로딩 ----------------------
def load_logs_parallel(log_dir, num_threads):
    log_files = []
    pattern = re.compile(r"folio_stats_snapshot_(\d+)\.log")
    for entry in os.listdir(log_dir):
        full_path = os.path.join(log_dir, entry)
        if os.path.isfile(full_path):
            if pattern.fullmatch(entry):
                log_files.append(full_path)
    log_files.sort()  # 알파벳 순 정렬 (파일명에 숫자가 포함되어 있다면 올바른 순서여야 함)

    def worker(tid):
        # 각 스레드는 자신 인덱스(tid)부터 num_threads 간격으로 파일 처리
        for i in range(tid, len(log_files), num_threads):
            fname = os.path.basename(log_files[i])
            m = pattern.fullmatch(fname)
            if m:
                snapshot_idx = int(m.group(1))
                parse_log_file(log_files[i], snapshot_idx)

    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(f"Loaded {len(log_files)} snapshot logs.")

# ---------------------- 노드별 히트맵 시각화 ----------------------
def visualize_heatmap_node(node, matrix):
    plt.figure()
    # 이미 matrix는 NumPy 배열이므로 np.array() 호출은 생략할 수 있음
    plt.imshow(matrix, aspect="auto", origin="lower")
    plt.xlabel("Snapshot Index")
    plt.ylabel("PFN Index (relative to node)")
    plt.title(f"Folio Migration Heatmap - Node {node}")
    plt.colorbar()
    filename = f"heatmap_node_{node}.png"
    plt.savefig(filename)
    plt.close()  # 사용한 Figure 자원 해제
    print(f"Saved heatmap for node {node} as {filename}")

# ---------------------- 메인 함수 ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize heatmap from folio stats logs."
    )
    parser.add_argument(
        "-d",
        "--directory",
        required=True,
        help="Directory containing log files (folio_stats_snapshot_*.log)",
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=1, help="Number of threads to use"
    )
    args = parser.parse_args()

    log_dir = args.directory
    num_threads = args.threads

    pfn_stats_file = "/sys/kernel/debug/numa_folio/pfn_stats"

    # 1. PFN 범위(노드별) 로딩
    node_ranges = []
    print(f"Loading PFN stats from {pfn_stats_file}...")
    load_pfn_ranges_stats(pfn_stats_file, node_ranges)
    if not node_ranges:
        print("Error: No node ranges loaded. Exiting.")
        sys.exit(1)

    # 노드 번호 기준으로 정렬
    node_ranges.sort(key=lambda nr: nr.node)

    # 2. Snapshot 로그 파일 병렬 로딩
    print(f"Loading logs from {log_dir} using {num_threads} thread(s)...")
    load_logs_parallel(log_dir, num_threads)

    # 3. 전체 스냅샷 개수 계산 (최대 snapshot index + 1)
    num_snapshots = max(snapshot_data.keys()) + 1 if snapshot_data else 0

    # 4. 각 노드별 매트릭스 생성 (NumPy 배열 사용) 및 히트맵 시각화
    for nr in node_ranges:
        nRows = nr.end - nr.start + 1
        print(f"Building matrix for node {nr.node} with {nRows} rows and {num_snapshots} columns.")
        # NumPy를 사용하여 연속된 메모리 공간에 0으로 초기화된 행렬 생성
        matrix = np.zeros((nRows, num_snapshots), dtype=np.float64)

        # snapshot_data의 각 snapshot에 대해 vectorized하게 처리
        for snap_idx, stats in snapshot_data.items():
            if not stats: 
                continue
            # 각 snapshot의 PFN과 migrate_count를 NumPy 배열로 변환
            pfns = np.array([stat.pfn for stat in stats], dtype=np.uint64)
            counts = np.array([stat.migrate_count for stat in stats], dtype=np.float64)
            # 현재 노드의 범위 내에 있는 인덱스를 mask로 구함
            mask = (pfns >= nr.start) & (pfns <= nr.end)
            if np.any(mask):
                # 해당 스냅샷에서 조건을 만족하는 PFN값들의 상대 인덱스를 계산
                row_indices = pfns[mask] - nr.start
                # vectorized 업데이트: 각 row_indices에 대해 해당 snapshot 열에 migrate_count 업데이트
                matrix[row_indices, snap_idx] = counts[mask]
        print(f"Visualizing heatmap for node {nr.node}...")
        visualize_heatmap_node(nr.node, matrix)

if __name__ == "__main__":
    main()
