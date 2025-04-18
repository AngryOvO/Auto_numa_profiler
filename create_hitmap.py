import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_node_heatmaps(csv_file, output_dir):
    # CSV 파일 읽기
    data = pd.read_csv(csv_file)

    # 노드별로 데이터 분리
    nodes = data['node'].unique()  # 노드 번호 추출
    for node in nodes:
        # 해당 노드의 데이터 필터링
        node_data = data[data['node'] == node]

        # Pivot 테이블 생성 (x축: snapshot, y축: pfn, 값: migrate_count)
        pivot = node_data.pivot_table(index='pfn', columns='snapshot', values='migrate_count', aggfunc='sum', fill_value=0)

        # 히트맵 생성
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, cmap="viridis", cbar=True)
        plt.title(f"NUMA Folio Migration Heatmap (Node {node})")
        plt.xlabel("Snapshot")
        plt.ylabel("PFN")
        plt.tight_layout()

        # 히트맵 저장
        output_file = f"{output_dir}/heatmap_node_{node}.png"
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Heatmap for Node {node} saved to {output_file}")

if __name__ == "__main__":
    csv_file = "/mnt/tmpfs/integrated_data.csv"  # CSV 파일 경로
    output_dir = "./heatmaps"  # 히트맵 저장 디렉토리

    # 히트맵 생성 함수 호출
    generate_node_heatmaps(csv_file, output_dir)