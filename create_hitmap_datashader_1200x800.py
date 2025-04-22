#!/usr/bin/env python3
import os
import sys
import re
import argparse
import threading
from dataclasses import dataclass
import numpy as np

# Dask와 Datashader 관련 임포트
import dask.array as da
import xarray as xr
import datashader.transfer_functions as tf
from datashader.utils import export_image
import datashader as ds  # Canvas를 사용하기 위해 추가

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
            i += 1  # 다음 줄이 범위 정보
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
                        pfn = int(tokens[0])
                        source_pfn = int(tokens[1])
                        migrate_count = int(tokens[2])
                        stats.append(FolioStat(pfn=pfn, source_pfn=source_pfn, migrate_count=migrate_count))
                    except Exception as e:
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
        if os.path.isfile(full_path) and pattern.fullmatch(entry):
            log_files.append(full_path)
    log_files.sort()  # 숫자에 따른 정렬

    def worker(tid):
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

# ---------------------- 노드별 히트맵 시각화 (1200x800 해상도로 다운샘플링) ----------------------
def visualize_heatmap_node_dask(node, matrix, num_threads):
    """
    NumPy 배열 matrix (nRows x num_snapshots)를 dask array로 변환한 후,
    xarray DataArray로 감싸 Datashader의 Canvas를 사용하여 1200x800 해상도로 aggregation(집계)합니다.
    그 후 shading을 하고 export_image()를 사용해 PNG 파일로 저장합니다.
    """
    # 목표 해상도 지정 (출력 이미지 픽셀 수)
    output_width = 1200
    output_height = 800
    nRows, nCols = matrix.shape

    # 데이터를 효율적으로 처리하기 위해 dask array로 변환 (청크 크기는 num_threads를 이용)
    dask_matrix = da.from_array(matrix, chunks=(max(1, nRows // num_threads), nCols))
    # xarray DataArray 생성; dims를 ["y", "x"]로 지정하면 y: PFN index, x: snapshot index
    xr_da = xr.DataArray(dask_matrix, dims=["y", "x"])

    # Datashader의 Canvas를 생성하여 출력 해상도와 범위 지정
    cvs = ds.Canvas(plot_width=output_width, plot_height=output_height,
                    x_range=(0, nCols), y_range=(0, nRows))
    # raster()를 사용해 원본 데이터를 1200x800 해상도로 aggregation합니다.
    agg = cvs.raster(xr_da)
    # shading: aggregation된 값을 색상으로 매핑 (여기서는 선형 보간법 사용)
    shaded = tf.shade(agg, how="linear", cmap=["lightblue", "darkblue"])
    # export_image를 사용하여 바로 PNG 파일로 저장.
    # fmt에 선행 점을 포함하여 최종 파일명이 "heatmap_node_{node}.png"가 되도록 함.
    filename = f"heatmap_node_{node}"
    export_image(shaded, filename=filename, fmt=".png")
    print(f"Saved aggregated heatmap for node {node} as {filename}.png")

# ---------------------- 메인 함수 ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize heatmap from folio stats logs using Datashader and Dask, aggregated to 1200x800 resolution."
    )
    parser.add_argument(
        "-d", "--directory", required=True,
        help="Directory containing log files (folio_stats_snapshot_*.log)"
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=1,
        help="Number of threads to use"
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
    node_ranges.sort(key=lambda nr: nr.node)

    # 2. Snapshot 로그 파일 병렬 로딩
    print(f"Loading logs from {log_dir} using {num_threads} thread(s)...")
    load_logs_parallel(log_dir, num_threads)

    # 3. 전체 스냅샷 개수 계산 (최대 snapshot index + 1)
    num_snapshots = max(snapshot_data.keys()) + 1 if snapshot_data else 0

    # 4. 각 노드별 매트릭스 생성 및 히트맵 시각화
    for nr in node_ranges:
        nRows = nr.end - nr.start + 1
        print(f"Building matrix for node {nr.node} with {nRows} rows and {num_snapshots} columns.")
        # NumPy 배열 생성 (행: PFN 범위 길이, 열: 스냅샷 수)
        matrix = np.zeros((nRows, num_snapshots), dtype=np.float64)

        # 각 스냅샷의 데이터를 vectorized하게 업데이트
        for snap_idx, stats in snapshot_data.items():
            if not stats:
                continue
            pfns = np.array([stat.pfn for stat in stats], dtype=np.uint64)
            counts = np.array([stat.migrate_count for stat in stats], dtype=np.float64)
            mask = (pfns >= nr.start) & (pfns <= nr.end)
            if np.any(mask):
                row_indices = pfns[mask] - nr.start
                matrix[row_indices, snap_idx] = counts[mask]
        print(f"Aggregating and generating heatmap for node {nr.node} at 1200x800 resolution...")
        visualize_heatmap_node_dask(nr.node, matrix, num_threads)

if __name__ == "__main__":
    main()
