#include <matplot/matplot.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <limits>
#include <algorithm>

using namespace std;
using namespace matplot;

// CSV 파일의 한 행을 저장할 구조체
struct DataRow {
    int snapshot;
    int node;
    int pfn;
    int source_nid;
    int migrate_count;
};

// 노드의 PFN 범위를 저장할 구조체
struct NodeRange {
    int start_pfn;
    int end_pfn;
};

int main() {
    // 1. PFN 범위 파일 읽기: /sys/kernel/debug/numa_folio/pfn_stats
    ifstream pfn_file("/sys/kernel/debug/numa_folio/pfn_stats");
    if (!pfn_file.is_open()) {
        cerr << "PFN 범위 파일을 열 수 없습니다." << endl;
        return 1;
    }

    string line;
    map<int, NodeRange> node_ranges;
    int current_node = -1;  // 현재 노드 ID 저장
    while (getline(pfn_file, line)) {
        if (line.empty())
            continue;

        // "node X" 형식의 행 처리
        if (line.find("node") != string::npos) {
            stringstream ss(line);
            string dummy;
            ss >> dummy >> current_node;  // 예: "node 0" → current_node에 0 저장
        }
        // "start pfn:" 행 처리
        else if (line.find("start pfn:") != string::npos) {
            size_t pos = line.find(":");
            int start_pfn = stoi(line.substr(pos + 1));
            node_ranges[current_node].start_pfn = start_pfn;
        }
        // "end pfn:" 행 처리
        else if (line.find("end pfn:") != string::npos) {
            size_t pos = line.find(":");
            int end_pfn = stoi(line.substr(pos + 1));
            node_ranges[current_node].end_pfn = end_pfn;
        }
    }
    pfn_file.close();

    cout << "노드별 PFN 범위:" << endl;
    for (const auto& entry : node_ranges) {
        cout << "Node " << entry.first << ": " 
             << entry.second.start_pfn << " ~ " 
             << entry.second.end_pfn << endl;
    }

    // 2. CSV 파일 읽기: integrated_data.csv
    // CSV 파일 첫 줄은 "snapshot, node, pfn, source_nid, migrate_count" 형식임
    ifstream csv_file("integrated_data.csv");
    if (!csv_file.is_open()) {
        cerr << "CSV 파일을 열 수 없습니다." << endl;
        return 1;
    }
    
    // 헤더 줄 건너뛰기
    getline(csv_file, line);
    
    vector<DataRow> data;
    while (getline(csv_file, line)) {
        if (line.empty())
            continue;
        stringstream ss(line);
        string token;
        DataRow row;
        
        // 순서대로 snapshot, node, pfn, source_nid, migrate_count 파싱
        if (getline(ss, token, ',')) {
            row.snapshot = stoi(token);
        }
        if (getline(ss, token, ',')) {
            row.node = stoi(token);
        }
        if (getline(ss, token, ',')) {
            row.pfn = stoi(token);
        }
        if (getline(ss, token, ',')) {
            row.source_nid = stoi(token);
        }
        if (getline(ss, token, ',')) {
            row.migrate_count = stoi(token);
        }
        data.push_back(row);
    }
    csv_file.close();

    cout << "\nCSV 데이터 샘플:" << endl;
    for (size_t i = 0; i < min(data.size(), size_t(5)); i++) {
        cout << data[i].snapshot << ", " << data[i].node << ", " 
             << data[i].pfn << ", " << data[i].source_nid 
             << ", " << data[i].migrate_count << endl;
    }

    // 3. 전체 snapshot 범위 산출 (CSV 데이터에서의 최소, 최대 snapshot 값)
    int snapshot_min = numeric_limits<int>::max();
    int snapshot_max = numeric_limits<int>::min();
    for (const auto& row : data) {
        snapshot_min = min(snapshot_min, row.snapshot);
        snapshot_max = max(snapshot_max, row.snapshot);
    }
    int num_snapshots = snapshot_max - snapshot_min + 1;

    // 4. 노드별로 PFN 범위를 기준으로 피벗 테이블(2차원 배열) 구성 및 히트맵 생성
    // 피벗 테이블은 해당 노드의 전체 PFN 범위(시작 ~ 끝)를 모두 포함하며,
    // CSV 데이터에 없는 PFN은 기본값 0으로 채워집니다.
    for (const auto& entry : node_ranges) {
        int node_id = entry.first;
        int start_pfn = entry.second.start_pfn;
        int end_pfn = entry.second.end_pfn;
        int num_pfns = end_pfn - start_pfn + 1;

        // PFN (행) x snapshot (열) 구조의 2차원 배열을 0으로 초기화
        vector<vector<double>> pivot(num_pfns, vector<double>(num_snapshots, 0.0));

        // CSV 데이터를 순회하며 해당 노드에 대한 값을 누적 (없으면 그대로 0)
        for (const auto& row : data) {
            if (row.node != node_id)
                continue;
            if (row.pfn < start_pfn || row.pfn > end_pfn)
                continue;
            int row_idx = row.pfn - start_pfn;       // PFN 인덱스
            int col_idx = row.snapshot - snapshot_min; // snapshot 인덱스
            pivot[row_idx][col_idx] += row.migrate_count;
        }

        // 5. matplot++로 히트맵 생성 및 저장 (노드별)
        auto fig = figure();
        imagesc(pivot);
        title("Node " + to_string(node_id) + " - Migration Heatmap");
        xlabel("Snapshot (Time)");
        ylabel("PFN");
        colormap("hot"); // "hot" colormap 사용

        // PNG 파일 저장 (예: node_0_migration_heatmap.png)
        string filename = "node_" + to_string(node_id) + "_migration_heatmap.png";
        save(filename);
        close(); // 현재 Figure 닫기

        cout << "노드 " << node_id << "의 히트맵이 '" << filename << "'에 저장되었습니다." << endl;
    }

    return 0;
}
