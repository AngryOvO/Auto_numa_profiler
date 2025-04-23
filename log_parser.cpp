// log_parser.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <regex>
#include <vector>
#include <string>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include <thread>
#include <mutex>

namespace py = pybind11;
namespace fs = std::filesystem;

// SnapshotMap: key는 snapshot index, value는 (pfn, source_pfn, migrate_count) 튜플의 벡터
using SnapshotMap = std::unordered_map<int, std::vector<std::tuple<int, int, int>>>;

SnapshotMap parse_logs(const std::string &log_dir, int num_threads) {
    // 1. 로그 파일 목록 수집
    std::vector<fs::path> log_files;
    for (const auto &entry : fs::directory_iterator(log_dir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            // 파일명이 "folio_stats_snapshot_"로 시작하고 ".log" 확장자인지 확인
            if (filename.rfind("folio_stats_snapshot_", 0) == 0 && entry.path().extension() == ".log") {
                log_files.push_back(entry.path());
            }
        }
    }
    std::sort(log_files.begin(), log_files.end());

    SnapshotMap result;
    std::mutex result_mutex;
    std::regex pattern(R"((\d+),\s*(\d+),\s*(\d+))");

    // worker lambda: [start, end) 범위의 파일들을 파싱
    auto worker = [&](int start, int end) {
        for (int i = start; i < end; i++) {
            const auto &file_path = log_files[i];
            // 파일 이름에서 snapshot index 추출 
            // 예: "folio_stats_snapshot_42.log" → snapshot index = 42
            std::string fname = file_path.filename().string();
            int snapshot_idx = 0;
            try {
                size_t prefix_len = std::string("folio_stats_snapshot_").size();
                size_t pos = fname.find(".log", prefix_len);
                std::string num_str = fname.substr(prefix_len, pos - prefix_len);
                snapshot_idx = std::stoi(num_str);
            } catch (...) {
                continue;  // 변환에 실패하면 건너뜁니다.
            }
            std::ifstream infile(file_path);
            if (!infile.is_open())
                continue;
            std::vector<std::tuple<int, int, int>> stats;
            std::string line;
            while (std::getline(infile, line)) {
                if (line.empty())
                    continue;
                std::smatch match;
                if (std::regex_search(line, match, pattern)) {
                    int pfn = std::stoi(match[1].str());
                    int source_pfn = std::stoi(match[2].str());
                    int migrate_count = std::stoi(match[3].str());
                    stats.push_back(std::make_tuple(pfn, source_pfn, migrate_count));
                }
            }
            {
                std::lock_guard<std::mutex> lock(result_mutex);
                result[snapshot_idx] = stats;
            }
        }
    };

    // 2. 파일들을 num_threads 개의 그룹으로 분할하여 worker 스레드 생성
    int total_files = log_files.size();
    if (num_threads < 1) {
        num_threads = 1;
    }
    int files_per_thread = total_files / num_threads;
    int remainder = total_files % num_threads;
    std::vector<std::thread> threads;
    int start_index = 0;
    for (int t = 0; t < num_threads; t++) {
        int count = files_per_thread + (t < remainder ? 1 : 0);
        int end_index = start_index + count;
        threads.emplace_back(worker, start_index, end_index);
        start_index = end_index;
    }
    for (auto &th : threads) {
        th.join();
    }
    return result;
}

PYBIND11_MODULE(log_parser, m) {
    m.doc() = "Module for multithreaded log file parsing using C++ and pybind11";
    m.def("parse_logs", &parse_logs,
          "Parse snapshot log files from a given directory using multithreading",
          py::arg("log_dir"), py::arg("num_threads") = 4);
}
