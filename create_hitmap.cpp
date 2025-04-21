#include <iostream>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <vector>
#include <string>
#include <regex>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
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

// ---------------------- 글로벌 변수 ----------------------
std::unordered_map<uint64_t, int> pfn_to_row_index;
std::vector<uint64_t> pfn_list;  // row 순서 고정용
std::unordered_map<int, std::vector<FolioStat>> snapshot_data;

// ---------------------- PFN 범위 로딩 ----------------------
void load_pfn_ranges(const std::string& filename) {
    std::ifstream infile(filename);
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        int node;
        uint64_t start, end;
        if (iss >> node >> start >> end) {
            for (uint64_t pfn = start; pfn <= end; ++pfn) {
                pfn_to_row_index[pfn] = pfn_list.size();
                pfn_list.push_back(pfn);
            }
        }
    }
    std::cout << "PFN ranges loaded. Total tracked PFNs: " << pfn_list.size() << "\n";
}

// ---------------------- 로그 파일 파싱 ----------------------
void parse_log_file(const fs::path& filepath, int snapshot_index) {
    std::ifstream file(filepath);
    std::string line;
    std::vector<FolioStat> stats;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ',')) tokens.push_back(token);

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

// ---------------------- 병렬 로딩 ----------------------
void load_logs_parallel(const std::string& log_dir, int num_threads) {
    std::vector<std::thread> threads;
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

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }

    for (auto& t : threads) t.join();

    std::cout << "Loaded " << log_files.size() << " snapshot logs.\n";
}

// ---------------------- 히트맵 데이터 생성 ----------------------
std::vector<std::vector<int>> build_heatmap_matrix() {
    size_t num_pfns = pfn_list.size();
    size_t num_snapshots = snapshot_data.size();
    std::vector<std::vector<int>> matrix(num_pfns, std::vector<int>(num_snapshots, 0));

    for (const auto& [snap_idx, stats] : snapshot_data) {
        for (const auto& stat : stats) {
            if (pfn_to_row_index.count(stat.pfn)) {
                size_t row = pfn_to_row_index[stat.pfn];
                matrix[row][snap_idx] = stat.migrate_count;
            }
        }
    }

    return matrix;
}

// ---------------------- 히트맵 시각화 ----------------------
void visualize_heatmap(const std::vector<std::vector<int>>& matrix) {
    // 히트맵 생성
    auto h = heatmap(matrix);
    colormap(palette::hot());
    xlabel("Snapshot Index");
    ylabel("PFN Index");
    title("Folio Migration Heatmap");
    colorbar();
    
    // PNG 파일로 저장
    h->save("heatmap.png");
}


// ---------------------- 사용법 출력 ----------------------
void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [-d|--directory <log_dir>] [-t|--threads <num_threads>]\n";
}

// ---------------------- 메인 ----------------------
int main(int argc, char* argv[]) {
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
        std::cerr << "❗ Error: log directory not specified.\n";
        print_usage(argv[0]);
        return 1;
    }

    std::string pfn_file = "/sys/kernel/debug/numa_folio/pfn_node";

    std::cout << "PFN 범위 로딩 중..." << std::endl;
    load_pfn_ranges(pfn_file);

    std::cout << "로그 병렬 로딩 중..." << std::endl;
    load_logs_parallel(log_dir, num_threads);

    std::cout << "행렬 생성 중..." << std::endl;
    auto matrix = build_heatmap_matrix();

    std::cout << "히트맵 시각화 중..." << std::endl;
    visualize_heatmap(matrix);

    return 0;
}
