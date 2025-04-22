import os
import re
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import sys


# ---------------------- 구조체 정의 ----------------------
class FolioStat:
    def __init__(self, pfn, source_pfn, migrate_count):
        self.pfn = pfn
        self.source_pfn = source_pfn
        self.migrate_count = migrate_count


class NodeRange:
    def __init__(self, node, start, end):
        self.node = node
        self.start = start
        self.end = end


# ---------------------- 글로벌 변수 ----------------------
# 스냅샷 인덱스 -> 해당 스냅샷의 FolioStat 목록
snapshot_data = {}


# ---------------------- PFN 범위 로딩 (pfn_stats 파일) ----------------------
def load_pfn_ranges_stats(filename):
    node_ranges = []
    try:
        with open(filename, 'r') as infile:
            lines = infile.readlines()

        idx = 0
        while idx < len(lines):
            line = lines[idx]
            node_match = re.match(r"node\s+(\d+)", line)
            if node_match:
                node = int(node_match.group(1))
                idx += 1
                range_line = lines[idx]
                range_match = re.match(r"start pfn:\s*(\d+),\s*end pfn:\s*(\d+)", range_line)
                if range_match:
                    start = int(range_match.group(1))
                    end = int(range_match.group(2))
                    node_ranges.append(NodeRange(node, start, end))
                    print(f"Loaded node range: node {node} start: {start} end: {end}")
            idx += 1
    except FileNotFoundError:
        print(f"Error: Unable to open pfn_stats file: {filename}")
    return node_ranges


# ---------------------- 로그 파일 파싱 ----------------------
def parse_log_file(filepath, snapshot_index):
    stats = []
    try:
        with open(filepath, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 3:
                    pfn = int(row[0])
                    source_pfn = int(row[1])
                    migrate_count = int(row[2])
                    stats.append(FolioStat(pfn, source_pfn, migrate_count))
        snapshot_data[snapshot_index] = stats
    except Exception as e:
        print(f"Error: Could not open or parse file {filepath}: {e}")


# ---------------------- 병렬 로그 로딩 ----------------------
def load_logs_parallel(log_dir, num_threads):
    log_files = []
    pattern = re.compile(r"folio_stats_snapshot_(\d+)\.log")

    for entry in os.scandir(log_dir):
        if entry.is_file() and pattern.match(entry.name):
            log_files.append(entry.path)  # 수정된 부분: entry.path() -> entry.path

    log_files.sort()  # 시간 순 정렬

    def worker(thread_id):
        for i in range(thread_id, len(log_files), num_threads):
            filename = log_files[i]
            match = pattern.match(os.path.basename(filename))
            if match:
                snapshot_index = int(match.group(1))
                parse_log_file(filename, snapshot_index)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i in range(num_threads):
            executor.submit(worker, i)

    print(f"Loaded {len(log_files)} snapshot logs.")



# ---------------------- 노드별 히트맵 시각화 ----------------------
def visualize_heatmap_node(node, matrix):
    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix, cmap="YlGnBu", cbar=True)
    plt.xlabel("Snapshot Index")
    plt.ylabel("PFN Index (relative to node)")
    plt.title(f"Folio Migration Heatmap - Node {node}")
    filename = f"heatmap_node_{node}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved heatmap for node {node} as {filename}")


# ---------------------- 사용법 출력 ----------------------
def print_usage():
    print("Usage: python script.py [-d|--directory <log_dir>] [-t|--threads <num_threads>]")


# ---------------------- 메인 ----------------------
if __name__ == "__main__":
    log_dir = ""
    num_threads = 1

    # 명령행 인자 처리
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] in ['-d', '--directory']:
            log_dir = args[i + 1]
            i += 2
        elif args[i] in ['-t', '--threads']:
            num_threads = int(args[i + 1])
            i += 2
        else:
            print_usage()
            sys.exit(1)

    if not log_dir:
        print("Error: log directory not specified.")
        print_usage()
        sys.exit(1)

    pfn_stats_file = "/sys/kernel/debug/numa_folio/pfn_stats"

    # 1. PFN 범위(노드별) 로딩
    print(f"Loading PFN stats from {pfn_stats_file}...")
    node_ranges = load_pfn_ranges_stats(pfn_stats_file)
    if not node_ranges:
        print("Error: No node ranges loaded. Exiting.")
        sys.exit(1)

    # 노드들을 0번부터 순서대로 처리하기 위해 정렬 (node 값 기준)
    node_ranges.sort(key=lambda x: x.node)

    # 2. Snapshot 로그 파일 병렬 로딩
    print(f"Loading logs from {log_dir} using {num_threads} thread(s)...")
    load_logs_parallel(log_dir, num_threads)

    # 3. 전체 스냅샷 개수 계산 (최대 snapshot index + 1)
    num_snapshots = max(snapshot_data.keys(), default=0) + 1

    # 4. 각 노드별로 순차적으로 부하를 줄이기 위해 매트릭스 생성 -> 히트맵 시각화 -> 메모리 해제
    for nr in node_ranges:
        nRows = nr.end - nr.start + 1
        print(f"Building matrix for node {nr.node} with {nRows} rows and {num_snapshots} columns.")

        # 노드별 매트릭스 생성 (초기값 0)
        matrix = np.zeros((nRows, num_snapshots))

        # 스냅샷 데이터 업데이트: 각 snapshot의 모든 stat를 순회하여, 노드 범위 내이면 해당 셀 갱신
        for snap_idx, stats in snapshot_data.items():
            for stat in stats:
                if nr.start <= stat.pfn <= nr.end:
                    row_index = stat.pfn - nr.start
                    matrix[row_index, snap_idx] = stat.migrate_count

        # 해당 노드의 히트맵 시각화 (파일로 저장)
        print(f"Visualizing heatmap for node {nr.node}...")
        visualize_heatmap_node(nr.node, matrix)

    print("Finished processing all nodes.")
