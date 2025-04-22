#include <iostream>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <vector>
#include <string>
#include <regex>
#include <thread>
#include <sstream>
#include <algorithm>
#include <getopt.h>
#include <matplot/matplot.h>

namespace fs = std::filesystem;
using namespace matplot;

// ---------------------- 구조체 정의 ----------------------
struct FolioStat {
    uint64_t pfn;
    uint64_t source_pfn;
    int migrate_count;
};

struct NodeRange {
    int node;
    uint64_t start;
    uint64_t end;
};

// ---------------------- 글로벌 변수 ----------------------
// 스냅샷 인덱스 -> 해당 스냅샷의 FolioStat 목록
std::unordered_map<int, std::vector<FolioStat>> snapshot_data;

// ---------------------- PFN 범위 로딩 (pfn_stats 파일) ----------------------
void load_pfn_ranges_stats(const std::string& filename, std::vector<NodeRange>& node_ranges) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error: Unable to open pfn_stats file: " << filename << "\n";
        return;
    }
    std::string line;
    // 파일은 "node X" 라는 줄과 이어서 "start pfn: Y, end pfn: Z"라는 줄이 번갈아 나오도록 가정
    while (std::getline(infile, line)) {
        std::regex node_regex(R"(node\s+(\d+))");
        std::smatch match;
        if (std::regex_search(line, match, node_regex)) {
            int node = std::stoi(match[1].str());
            if (std::getline(infile, line)) {
                std::regex range_regex(R"(start pfn:\s*(\d+),\s*end pfn:\s*(\d+))");
                std::smatch range_match;
                if (std::regex_search(line, range_match, range_regex)) {
                    uint64_t start = std::stoull(range_match[1].str());
                    uint64_t end = std::stoull(range_match[2].str());
                    node_ranges.push_back({node, start, end});
                    std::cout << "Loaded node range: node " << node 
                              << " start: " << start << " end: " << end << "\n";
                } else {
                    std::cerr << "Error parsing range line: " << line << "\n";
                }
            }
        }
    }
}

// ---------------------- 로그 파일 파싱 ----------------------
void parse_log_file(const fs::path& filepath, int snapshot_index) {
    std::ifstream file(filepath);
    if (!file) {
        std::cerr << "Error: Could not open file " << filepath.string() << "\n";
        return;
    }
    std::string line;
    std::vector<FolioStat> stats;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() == 3) {
            FolioStat stat;
            stat.pfn = std::stoull(tokens[0]);
            stat.source_pfn = std::stoull(tokens[1]);
            stat.migrate_count = std::stoi(tokens[2]);
            stats.push_back(stat);
        }
    }
    snapshot_data[snapshot_index] = std::move(stats);
}

// ---------------------- 병렬 로그 로딩 ----------------------
void load_logs_parallel(const std::string& log_dir, int num_threads) {
    std::vector<fs::path> log_files;
    std::regex pattern(R"(folio_stats_snapshot_(\d+)\.log)");

    for (const auto& entry : fs::directory_iterator(log_dir)) {
        if (!entry.is_regular_file()) continue;
        std::string filename = entry.path().filename().string();
        if (std::regex_match(filename, pattern)) {
            log_files.push_back(entry.path());
        }
    }

    std::sort(log_files.begin(), log_files.end()); // 시간 순 정렬

    auto worker = [&](int tid) {
        for (size_t i = tid; i < log_files.size(); i += num_threads) {
            std::smatch match;
            std::string fname = log_files[i].filename().string();
            if (std::regex_match(fname, match, pattern)) {
                int snapshot_idx = std::stoi(match[1].str());
                parse_log_file(log_files[i], snapshot_idx);
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    for (auto& t : threads) {
        t.join();
    }
    std::cout << "Loaded " << log_files.size() << " snapshot logs.\n";
}

// ---------------------- 노드별 히트맵 시각화 ----------------------
#include <cstdlib>  // for setenv

void visualize_heatmap_node(int node, const std::vector<std::vector<double>>& matrix) {
    // Qt 그래픽 비활성화 (헤드리스 모드)
    setenv("QT_QPA_PLATFORM", "offscreen", 1);

    using namespace matplot;
    auto h = image(matrix, true);
    xlabel("Snapshot Index");
    ylabel("PFN Index (relative to node)");
    title("Folio Migration Heatmap - Node " + std::to_string(node));
    colorbar();

    // 현재 Figure 핸들을 얻어서 저장합니다.
    std::string filename = "heatmap_node_" + std::to_string(node) + ".png";
    save(filename);
    std::cout << "Saved heatmap for node " << node << " as " << filename << "\n";
}


// ---------------------- 사용법 출력 ----------------------
void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [-d|--directory <log_dir>] [-t|--threads <num_threads>]\n";
}

// ---------------------- 메인 ----------------------
int main(int argc, char* argv[]) {
    // 헤드리스 환경에서 offscreen 모드 사용
    setenv("QT_QPA_PLATFORM", "offscreen", 1);
    
    std::string log_dir;
    int num_threads = 1;

    const char* const short_opts = "d:t:";
    const option long_opts[] = {
        {"directory", required_argument, nullptr, 'd'},
        {"threads", required_argument, nullptr, 't'},
        {nullptr, 0, nullptr, 0}
    };

    while (true) {
        const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);
        if (opt == -1) break;

        switch (opt) {
            case 'd':
                log_dir = optarg;
                break;
            case 't':
                num_threads = std::stoi(optarg);
                break;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    if (log_dir.empty()) {
        std::cerr << "Error: log directory not specified.\n";
        print_usage(argv[0]);
        return 1;
    }

    std::string pfn_stats_file = "/sys/kernel/debug/numa_folio/pfn_stats";

    // 1. PFN 범위(노드별) 로딩
    std::vector<NodeRange> node_ranges;
    std::cout << "Loading PFN stats from " << pfn_stats_file << "...\n";
    load_pfn_ranges_stats(pfn_stats_file, node_ranges);
    if (node_ranges.empty()) {
        std::cerr << "Error: No node ranges loaded. Exiting.\n";
        return 1;
    }

    // 노드들을 0번부터 순서대로 처리하기 위해 정렬 (node 값 기준)
    std::sort(node_ranges.begin(), node_ranges.end(), [](const NodeRange& a, const NodeRange& b) {
        return a.node < b.node;
    });

    // 2. Snapshot 로그 파일 병렬 로딩
    std::cout << "Loading logs from " << log_dir << " using " << num_threads << " thread(s)...\n";
    load_logs_parallel(log_dir, num_threads);

    // 3. 전체 스냅샷 개수 계산 (최대 snapshot index + 1)
    int num_snapshots = 0;
    for (const auto& [snap_idx, stats] : snapshot_data) {
        num_snapshots = std::max(num_snapshots, snap_idx + 1);
    }

    // 4. 각 노드별로 순차적으로 부하를 줄이기 위해 매트릭스 생성 -> 히트맵 시각화 -> 메모리 해제
    for (const auto &nr : node_ranges) {
        size_t nRows = nr.end - nr.start + 1;
        std::cout << "Building matrix for node " << nr.node 
                  << " with " << nRows << " rows and " << num_snapshots << " columns.\n";

        // 노드별 매트릭스 생성 (초기값 0)
        std::vector<std::vector<double>> matrix(nRows, std::vector<double>(num_snapshots, 0.0));

        // 스냅샷 데이터 업데이트: 각 snapshot의 모든 stat를 순회하여, 노드 범위 내이면 해당 셀 갱신
        for (const auto &entry : snapshot_data) {
            int snap_idx = entry.first;
            const auto &stats = entry.second;
            for (const auto &stat : stats) {
                if (stat.pfn >= nr.start && stat.pfn <= nr.end) {
                    size_t row_index = stat.pfn - nr.start;  // 상대 인덱스 계산
                    matrix[row_index][snap_idx] = static_cast<double>(stat.migrate_count);
                }
            }
        }

        // 해당 노드의 히트맵 시각화 (파일로 저장)
        std::cout << "Visualizing heatmap for node " << nr.node << "...\n";
        visualize_heatmap_node(nr.node, matrix);
        // matrix는 여기서 자동으로 범위 해제됨 (스코프 종료)
    }

    return 0;
}