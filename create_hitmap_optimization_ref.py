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
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

# C++ 확장 모듈 임포트 (빌드한 모듈)
import log_parser

# ---------------------- 구조체 정의 (dataclass 사용) ----------------------
@dataclass
class FolioStat:
    pfn: int
    refcount: int

@dataclass
class NodeRange:
    node: int
    start: int
    end: int

# ---------------------- 글로벌 변수 ----------------------
# (이전 snapshot_data 딕셔너리 대신 전역 배열를 사용함)
# ---------------------------------------------------------------------
# PFN 범위 로딩 (pfn_stats 파일)
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

# ---------------------- C++ 확장 모듈을 사용한 로그 파싱 (노드0 전역 배열 방식) ----------------------
def load_global_snapshot_array(log_dir, num_threads):
    """
    C++ 확장 모듈의 parse_logs_global_array() 함수를 호출하여,
    노드 0의 PFN 범위에 따른 전역 NumPy 배열을 반환한다.
    전역 배열의 shape는 (total_pfns, num_snapshots)이며, 각각의 셀에는 해당 PFN의 refcount 값이 기록된다.
    """
    print(f"Using C++ extension (global array) to parse logs in {log_dir} with {num_threads} threads...")
    global_array = log_parser.parse_logs_global_array(log_dir, num_threads)
    print(f"Global array shape: {global_array.shape}")
    return global_array

# ---------------------- 노드별 히트맵 시각화 (1200x800 해상도, Dask+Datashader) ----------------------
def visualize_heatmap_node_dask(nr, matrix, num_threads, global_vmax):
    """
    nr: NodeRange 객체 (노드 번호와 해당 노드의 PFN 범위)
    matrix: 해당 노드의 NumPy 배열 (nRows x num_snapshots)
    num_threads: 사용 스레드 수
    global_vmax: 모든 노드에서의 최대 reference count (모든 히트맵에 동일하게 적용)
    """
    node = nr.node
    output_width = 1200
    output_height = 800
    nRows, nCols = matrix.shape

    # Dask array 변환 및 xarray DataArray 생성 후 계산
    dask_matrix = da.from_array(matrix, chunks=(max(1, nRows // num_threads), nCols))
    xr_da = xr.DataArray(dask_matrix, dims=["y", "x"])
    computed_xr_da = xr_da.compute()

    # Datashader Canvas 생성 및 aggregation 수행
    cvs = ds.Canvas(plot_width=output_width, plot_height=output_height,
                    x_range=(0, nCols), y_range=(0, nRows))
    agg = cvs.raster(computed_xr_da)

    # vmin, vmax 설정
    vmin = 0
    vmax = global_vmax

    # 사용자 정의 colormap: navy → red → yellow
    colors = ["navy", "red", "yellow"]
    thermal_cmap = LinearSegmentedColormap.from_list("thermal", colors, N=256)

    plt.figure(figsize=(12, 8))  # 12x8 인치 → 1200x800 픽셀
    extent = (0, nCols, 0, nRows)
    img = plt.imshow(agg.values, cmap=thermal_cmap, origin="lower",
                     extent=extent, aspect="auto", vmin=vmin, vmax=vmax)
    plt.xlabel("Snapshot (Time)")
    plt.ylabel("PFN")
    plt.title(f"Node {node} - Reference Count Heatmap")
    
    # y축: 하단은 PFN start, 상단은 PFN end로 표시
    plt.yticks([0, nRows], [nr.start, nr.end])
    
    # 색상바: 정수 tick만 표시
    cbar = plt.colorbar(img, ticks=np.arange(vmin, vmax + 1))
    cbar.set_label("Reference Count")
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
    parser.add_argument("-t", "--threads", type=int, default=4,
                        help="Number of threads to use for log parsing")
    args = parser.parse_args()

    log_dir = args.directory
    num_threads = args.threads
    pfn_stats_file = "/sys/kernel/debug/numa_folio/pfn_stats"

    # PFN 범위 로딩 (모든 노드 정보 로드)
    node_ranges = []
    print(f"Loading PFN stats from {pfn_stats_file}...")
    load_pfn_ranges_stats(pfn_stats_file, node_ranges)
    if not node_ranges:
        print("Error: No node ranges loaded. Exiting.")
        sys.exit(1)
    node_ranges.sort(key=lambda nr: nr.node)

    # C++ 확장을 이용해 전역 배열(노드0 기준)을 파싱
    global_array = load_global_snapshot_array(log_dir, num_threads)

    # 전체 전역 배열에서 최대 reference count를 구함
    global_max = int(np.max(global_array))
    print(f"Global maximum reference count: {global_max}")

    # 새로운 방식은 C++ 확장이 노드 0 전용 배열을 반환하므로,
    # node_ranges 중에서 node 0에 해당하는 정보만 사용합니다.
    for nr in node_ranges:
        if nr.node != 0:
            print(f"Skipping node {nr.node} (global array is for node 0 only).")
            continue
        # global_array는 이미 노드 0 전체 PFN 정보를 담고있으므로 직접 사용
        matrix = np.array(global_array)  # matrix shape: (nRows, num_snapshots)
        print(f"Aggregating and generating heatmap for node {nr.node} ({nr.start} to {nr.end})...")
        visualize_heatmap_node_dask(nr, matrix, num_threads, global_max)

if __name__ == "__main__":
    main()
