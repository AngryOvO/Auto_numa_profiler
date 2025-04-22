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
import datashader as ds
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
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

# ---------------------- 노드별 히트맵 시각화 (1200x800 해상도로 다운샘플링 및 색상표 포함) ----------------------
def visualize_heatmap_node_dask(node, matrix, num_threads):
    """
    NumPy 배열 matrix (nRows x num_snapshots)를 Dask 배열로 변환한 후,
    xarray DataArray로 감싼 다음, Datashader의 Canvas를 사용해 1200x800 해상도로 aggregation(집계)합니다.
    그 후 aggregation 결과를 matplotlib로 표시하며,
      - vmin은 0으로 고정 (네이비색상),
      - vmax는 matrix의 최대 migration count (노란색),
      - BoundaryNorm을 통해 정수 값만 구분하는 colormap으로 설정합니다.
    """
    output_width = 1200
    output_height = 800
    nRows, nCols = matrix.shape

    # Dask array와 xarray DataArray 생성 및 바로 계산
    dask_matrix = da.from_array(matrix, chunks=(max(1, nRows // num_threads), nCols))
    xr_da = xr.DataArray(dask_matrix, dims=["y", "x"])
    computed_xr_da = xr_da.compute()

    # Datashader Canvas 생성 및 집계(aggregation)
    cvs = ds.Canvas(plot_width=output_width, plot_height=output_height,
                    x_range=(0, nCols), y_range=(0, nRows))
    agg = cvs.raster(computed_xr_da)

    # vmin, vmax 설정
    vmin = 0
    vmax = matrix.max()

    # 사용자 정의 colormap: 낮은 값은 네이비, 중간은 빨간색, 높은 값은 노란색
    colors = ["navy", "red", "yellow"]
    thermal_cmap = LinearSegmentedColormap.from_list("thermal", colors, N=256)

    # 정수 값에 맞춘 discrete한 색상 구분을 위해 BoundaryNorm 사용
    boundaries = np.arange(vmin, vmax + 2)  # 정수 경계: vmin부터 vmax+1까지
    norm = BoundaryNorm(boundaries, ncolors=thermal_cmap.N, clip=True)

    # matplotlib를 사용한 시각화: Figure 크기 12x8인치 → 1200x800 픽셀
    plt.figure(figsize=(12, 8))
    extent = (0, nCols, 0, nRows)
    img = plt.imshow(agg.values, cmap=thermal_cmap, norm=norm, origin="lower",
                 extent=extent, aspect="auto")
    plt.xlabel("Snapshot (Time)")
    plt.ylabel("PFN")
    plt.title(f"Node {node} - Migration Heatmap")
    # 색상바는 정수 tick만 사용하도록 설정
    cbar = plt.colorbar(img, ticks=np.arange(vmin, vmax + 1))
    cbar.set_label("Migration Count")
    plt.tight_layout()
    
    filename = f"heatmap_node_{node}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved aggregated heatmap for node {node} as {filename}")

# ---------------------- 메인 함수 ----------------------
def main():
    parser = argparse.ArgumentParser(
        description="Visualize heatmap from folio stats logs using Datashader and Dask, aggregated to 1200x800 resolution with a thermal style."
    )
    parser.add_argument("-d", "--directory", required=True,
                        help="Directory containing log files (folio_stats_snapshot_*.log)")
    parser.add_argument("-t", "--threads", type=int, default=1,
                        help="Number of threads to use")
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
        matrix = np.zeros((nRows, num_snapshots), dtype=np.float64)
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
