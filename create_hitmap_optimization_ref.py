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
            i += 1  # 다음 줄에 PFN 범위가 있음
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

# ---------------------- C++ 확장 모듈을 사용한 로그 파싱 (모든 노드 전역 배열 방식) ----------------------
def load_all_nodes_global_snapshot_array(log_dir, num_threads):
    """
    C++ 확장 모듈의 parse_logs_global_array_all_nodes() 함수를 호출하여,
    각 노드의 PFN 범위에 따른 전역 NumPy 배열 dict를 반환한다.
    반환 dict 형식: {node_id: NumPy array}, 각 배열의 shape는 (total_pfns, num_snapshots)
    """
    print(f"Using C++ extension (global array for all nodes) to parse logs in {log_dir} with {num_threads} threads...")
    global_arrays = log_parser.parse_logs_global_array_all_nodes(log_dir, num_threads)
    print(f"Global arrays returned for nodes: {list(global_arrays.keys())}")
    return global_arrays

# ---------------------- 노드별 히트맵 시각화 (사용자 지정 해상도, Dask+Datashader) ----------------------
def visualize_heatmap_node_dask(nr, matrix, num_threads, global_vmax, output_width, output_height):
    """
    nr: NodeRange 객체 (노드 번호와 해당 노드의 PFN 범위)
    matrix: 해당 노드의 NumPy 배열 (nRows x num_snapshots)
    num_threads: 사용 스레드 수
    global_vmax: 모든 노드에서의 최대 reference count (모든 히트맵에 동일하게 적용)
    output_width, output_height: 최종 출력 이미지의 해상도 (픽셀 단위)
    """
    node = nr.node
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

    # 사용자 정의 colormap: 0은 검은색, 1부터 navy → red → yellow
    colors = ["black", "navy", "red", "yellow", "white"]
    thermal_cmap = LinearSegmentedColormap.from_list("thermal", colors, N=256)

    # vmin, vmax 설정
    vmin = 1  # 최소값을 1로 변경하여 1부터 navy로 표현되도록 설정
    vmax = global_vmax

    plt.figure(figsize=figsize)
    extent = (0, nCols, 0, nRows)
    img = plt.imshow(agg.values, cmap=thermal_cmap, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax)

    # 0인 값은 검은색으로 표현되도록 설정
    thermal_cmap.set_under("black")

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
        description="Visualize heatmap from folio stats logs using Datashader and Dask, with custom resolution."
    )
    parser.add_argument("-d", "--directory", required=True,
                        help="Directory containing log files (folio_stats_snapshot_*.log)")
    parser.add_argument("-t", "--threads", type=int, default=4,
                        help="Number of threads to use for log parsing")
    parser.add_argument("--width", type=int, default=1200,
                        help="Output image width in pixels (default: 1200)")
    parser.add_argument("--height", type=int, default=800,
                        help="Output image height in pixels (default: 800)")
    args = parser.parse_args()

    log_dir = args.directory
    num_threads = args.threads
    output_width = args.width
    output_height = args.height
    pfn_stats_file = "/sys/kernel/debug/numa_folio/pfn_stats"

    # PFN 범위 로딩 (모든 노드 정보 로드)
    node_ranges = []
    print(f"Loading PFN stats from {pfn_stats_file}...")
    load_pfn_ranges_stats(pfn_stats_file, node_ranges)
    if not node_ranges:
        print("Error: No node ranges loaded. Exiting.")
        sys.exit(1)
    node_ranges.sort(key=lambda nr: nr.node)

    # C++ 확장을 이용해 모든 노드 전역 배열을 파싱
    global_arrays = load_all_nodes_global_snapshot_array(log_dir, num_threads)

    # 전체 전역 배열에서 최대 reference count를 구함 (모든 노드 배열 중 최대값)
    global_max = 0
    for arr in global_arrays.values():
        max_val = int(np.max(arr))
        if max_val > global_max:
            global_max = max_val
    print(f"Global maximum reference count: {global_max}")

    # 각 노드에 대해 히트맵 시각화 진행
    for nr in node_ranges:
        if nr.node not in global_arrays:
            print(f"No global array for node {nr.node}. Skipping.")
            continue
        matrix = np.array(global_arrays[nr.node])  # shape: (nRows, num_snapshots)
        print(f"Aggregating and generating heatmap for node {nr.node} ({nr.start} to {nr.end})...")
        visualize_heatmap_node_dask(nr, matrix, num_threads, global_max, output_width, output_height)

if __name__ == "__main__":
    main()
