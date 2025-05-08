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
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt

# C++ 확장 모듈 임포트 (빌드한 모듈)
import log_parser

# Numba 및 Datashader의 CustomReduction 임포트
import numba as nb
from datashader.reductions import CustomReduction

# ----------------------------------------
# 커스텀 aggregator: 각 픽셀 내 모든 값의 0,1,2에 대한 히스토그램을 누적한 후
# 최빈값(mode)을 선택하는 방식.
# 아래 함수들은 각각 초기화(init), 한 값(accumulate), 두 히스토그램 병합(merge),
# 그리고 최종 단계(finalize)를 구현합니다.
#
# 이 방식은 분산/병렬 집계에 적합하도록 설계되었습니다.
# ----------------------------------------

@nb.njit
def mode_init():
    # 히스토그램 초기화: 0, 1, 2 각 카테고리에 대해 3칸 배열 (int64)
    return np.zeros(3, dtype=np.int64)

@nb.njit
def mode_accumulate(hist, value):
    # value가 0,1,2 범위에 있다면 해당 bin을 1 증가시킴
    if value >= 0 and value < 3:
        hist[int(value)] += 1
    return hist

@nb.njit
def mode_merge(hist1, hist2):
    # 두 히스토그램을 element-wise로 합산
    return hist1 + hist2

@nb.njit
def mode_finalize(hist):
    # 최종 히스토그램에서 최대 카운트를 가진 인덱스를 반환 (최빈값)
    return np.argmax(hist)

# CustomReduction 객체 생성: 위의 네 함수를 전달합니다.
mode_reducer = CustomReduction(mode_init, mode_accumulate, mode_merge, mode_finalize)

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

# ---------------------- 노드별 히트맵 시각화 (사용자 지정 해상도, Dask+Datashader with 커스텀 aggregator) ----------------------
def visualize_heatmap_node_dask(nr, matrix, num_threads, global_vmax, output_width, output_height):
    """
    nr: NodeRange 객체 (노드 번호와 해당 노드의 PFN 범위)
    matrix: 해당 노드의 NumPy 배열 (nRows x num_snapshots)
    num_threads: 사용 스레드 수
    global_vmax: 모든 노드에서의 최대 reference count (이제는 2로 고정되어 있음)
    output_width, output_height: 최종 출력 이미지의 해상도 (픽셀 단위)
    """
    node = nr.node
    nRows, nCols = matrix.shape

    # Dask array 변환 및 xarray DataArray 생성 후 계산
    dask_matrix = da.from_array(matrix, chunks=(max(1, nRows // num_threads), nCols))
    xr_da = xr.DataArray(dask_matrix, dims=["y", "x"])
    computed_xr_da = xr_da.compute()

    # Datashader Canvas 생성 및 커스텀 mode_reducer를 적용하여 집계 수행
    cvs = ds.Canvas(plot_width=output_width, plot_height=output_height,
                    x_range=(0, nCols), y_range=(0, nRows))
    agg = cvs.raster(computed_xr_da, agg=mode_reducer)

    # --- 고정 컬러맵 구현 ---
    # 0: black, 1: navy, 2: red 로 매핑
    # 경계값은 -0.5, 0.5, 1.5, 2.5 로 설정하여 각 값이 정확히 해당 색상의 영역에 들어가게 함.
    boundaries = [-0.5, 0.5, 1.5, 2.5]
    color_list = ["black", "navy", "red"]
    listed_cmap = ListedColormap(color_list)
    norm = BoundaryNorm(boundaries, listed_cmap.N)

    # matplotlib figure 생성 (DPI 100 기준, 인치 단위)
    figsize = (output_width / 100, output_height / 100)
    plt.figure(figsize=figsize)
    extent = (0, nCols, 0, nRows)
    
    # imshow에 norm과 ListedColormap 지정
    img = plt.imshow(agg.values, cmap=listed_cmap, norm=norm, origin="lower",
                     extent=extent, aspect="auto")
    plt.xlabel("Snapshot (Time)")
    plt.ylabel("PFN")
    plt.title(f"Node {node} - Reference Count Heatmap")
    
    # y축에 PFN range 표시: 맨 아래는 nr.start, 맨 위는 nr.end
    plt.yticks([0, nRows], [nr.start, nr.end])
    
    # 색상바: 0, 1, 2 값을 tick으로 표시
    cbar = plt.colorbar(img, boundaries=boundaries, ticks=[0, 1, 2])
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

    # 모든 노드 배열에서 최대 refcount 값을 구함 (모든 노드 배열 중 최대값)
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
        matrix = np.array(global_arrays[nr.node])
        print(f"Aggregating and generating heatmap for node {nr.node} ({nr.start} to {nr.end})...")
        visualize_heatmap_node_dask(nr, matrix, num_threads, global_max, output_width, output_height)

if __name__ == "__main__":
    main()
