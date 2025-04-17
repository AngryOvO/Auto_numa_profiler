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
import multiprocessing as mp
from PIL import Image

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
            m = re.search(r'start pfn:?\s*(\d+)[,]?\s*(?:end pfn:?\s*(\d+))', line)
            if m and (current_node is not None):
                node_ranges[current_node] = (int(m.group(1)), int(m.group(2)))
                current_node = None
    return node_ranges

def generate_heatmap_chunk(args):
    """
    한 청크에 대해 히트맵 부분 이미지를 생성하는 함수.
    args: (node, chunk_id, pivot, start_idx, end_idx, global_vmax, cmap, output_dir)
    """
    node, chunk_id, pivot, start_idx, end_idx, global_vmax, cmap, output_dir = args
    # 청크에 해당하는 부분을 슬라이싱 (행: PFN 범위)
    pivot_chunk = pivot.loc[start_idx:end_idx]
    
    plt.figure(figsize=(10, 8))
    # imshow()는 전체 2차원 array를 한 번에 렌더링하므로 빠름.
    plt.imshow(pivot_chunk.values, aspect='auto', cmap=cmap, vmin=0, vmax=global_vmax)
    plt.title(f"Node {node} Chunk {chunk_id}")
    plt.xlabel("Snapshot (Time)")
    plt.ylabel("PFN")
    plt.colorbar()
    plt.tight_layout()
    
    out_file = os.path.join(output_dir, f"node_{node}_chunk_{chunk_id}.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Chunk {chunk_id} for node {node} saved as '{out_file}'.")
    return out_file

def stitch_images(image_files, output_file):
    """
    이미지 파일들을 세로로 결합하여 하나의 큰 이미지로 만든다.
    """
    images = [Image.open(f) for f in image_files]
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    
    max_width = max(widths)
    total_height = sum(heights)
    
    stitched = Image.new("RGB", (max_width, total_height))
    current_y = 0
    for img in images:
        stitched.paste(img, (0, current_y))
        current_y += img.height
    stitched.save(output_file)
    print(f"Stitched image saved as '{output_file}'.")

def combine_logs_and_generate_heatmaps(log_pattern="folio_stats_snapshot_*.log",
                                         node_pfn_stats_path="/sys/kernel/debug/numa_folio/pfn_stats"):
    """
    log_pattern에 맞는 모든 로그 파일을 결합하여 DataFrame을 생성하고,
    각 NUMA 노드별 히트맵을 생성하여 PNG 파일로 저장한다.
    
    (히트맵 생성 전 각 노드의 PFN 범위와 수집한 총 스냅샷 갯수를 출력한다.)
    그리고 한 노드의 히트맵 생성 작업은 PFN 행을 청크로 분할하여 병렬 처리한다.
    """
    # 로그 파일 목록 가져오기 (파일명에 포함된 스냅샷 번호 순으로 정렬)
    log_files = sorted(glob.glob(log_pattern),
                       key=lambda x: int(re.search(r"folio_stats_snapshot_(\d+)\.log", x).group(1)))
    if not log_files:
        print("No log files found matching pattern", log_pattern)
        sys.exit(1)

    collected_data = []
    # 로그 파일에서 folio 통계 라인을 추출하기 위한 정규식
    line_regex = re.compile(r"folio node:?\s*(\d+),\s*pfn:?\s*(\d+),\s*source_nid:?\s*(\d+),\s*migrate_count:?\s*(\d+)")
    
    # 각 로그 파일의 데이터 추출
    for log_file in log_files:
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
    
    # 노드별 PFN 범위 확보
    node_ranges = parse_node_pfn_stats(node_pfn_stats_path)
    print("Node PFN ranges:")
    print(node_ranges)
    
    # 스냅샷 범위 설정 (최소 ~ 최대)
    if df.empty:
        all_snapshots = range(0, 2)
    else:
        all_snapshots = range(df["snapshot"].min(), df["snapshot"].max() + 1)
    print("Total number of snapshots collected:", len(all_snapshots))
    
    # 전체 migrate_count의 최대값 (히트맵 색상 범위 기준)
    global_vmax = df["migrate_count"].max() if not df.empty else 1

    # 각 노드별 히트맵 생성 (x축: 스냅샷, y축: PFN)
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
            # 해당 노드의 전체 PFN 범위를 포함하도록 인덱스 재설정
            start_pfn, end_pfn = node_ranges[node]
            full_pfn_range = range(start_pfn, end_pfn + 1)
            pivot = pivot.reindex(index=full_pfn_range, fill_value=0)
        else:
            print(f"No migration data for node {node}. Generating empty heatmap.")
            start_pfn, end_pfn = node_ranges[node]
            full_pfn_range = range(start_pfn, end_pfn + 1)
            pivot = pd.DataFrame(0, index=full_pfn_range, columns=all_snapshots)

        print(f"Node {node} pivot shape: {pivot.shape}")

        # --- 병렬 처리로 한 노드 내 히트맵 생성 ---
        # PFN 범위를 일정 청크 단위로 분할
        chunk_size = 10000  # 필요에 따라 조정
        pivot_index = list(pivot.index)
        tasks = []
        chunk_id = 0
        current_index = 0
        # 각 청크별 작업 생성: pivot_index는 정렬되어 있으므로 순서대로 슬라이싱
        while current_index < len(pivot_index):
            start_idx = pivot_index[current_index]
            end_idx = pivot_index[min(current_index + chunk_size - 1, len(pivot_index) - 1)]
            tasks.append((node, chunk_id, pivot, start_idx, end_idx, global_vmax,
                          LinearSegmentedColormap.from_list("Thermal", ["navy", "red", "yellow"], N=256),
                          f"./chunks_node_{node}"))
            current_index += chunk_size
            chunk_id += 1

        # 해당 노드별 청크 임시 파일 저장 디렉터리 생성
        os.makedirs(f"./chunks_node_{node}", exist_ok=True)
        # 병렬 처리: 각 청크 이미지 생성
        with mp.Pool(processes=mp.cpu_count()) as pool:
            chunk_files = pool.map(generate_heatmap_chunk, tasks)
        # 청크 이미지들을 결합하여 최종 히트맵 이미지 생성
        final_filename = f"node_{node}_migration_heatmap.png"
        stitch_images(chunk_files, final_filename)
        print(f"Final heatmap for node {node} saved as '{final_filename}'.")
        # (선택사항) 임시 청크 파일 삭제 처리 가능

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
    # 결합된 데이터를 CSV 파일로 저장 (옵션)
    df.to_csv("combined_folio_stats.csv", index=False)
    print("Combined data saved as 'combined_folio_stats.csv'.")

if __name__ == "__main__":
    main()
