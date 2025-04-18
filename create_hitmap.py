import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_heatmap(csv_file, output_file):
    # CSV 파일 읽기
    data = pd.read_csv(csv_file)

    # Pivot 테이블 생성 (x축: snapshot, y축: pfn, 값: migrate_count)
    pivot = data.pivot_table(index='pfn', columns='snapshot', values='migrate_count', aggfunc='sum', fill_value=0)

    # 히트맵 생성
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, cmap="viridis", cbar=True)
    plt.title("NUMA Folio Migration Heatmap")
    plt.xlabel("Snapshot")
    plt.ylabel("PFN")
    plt.tight_layout()

    # 히트맵 저장
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Heatmap saved to {output_file}")

if __name__ == "__main__":
    csv_file = "integrated_data.csv"  # C 코드에서 생성된 CSV 파일
    output_file = "heatmap.png"      # 저장할 히트맵 이미지 파일
    generate_heatmap(csv_file, output_file)