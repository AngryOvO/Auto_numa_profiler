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
#include <algorithm>

namespace py = pybind11;
namespace fs = std::filesystem;

using SnapshotMap = std::unordered_map<int, std::vector<std::tuple<int, int, int>>>;

SnapshotMap parse_logs(const std::string &log_dir, int num_threads) {
    // 수집할 로그 파일 경로 목록
    std::vector<fs::path> log_files;
    for (const auto &entry : fs::directory_iterator(log_dir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.find("folio_stats_snapshot_") == 0 && entry.path().extension() == ".log")
                log_files.push_back(entry.path());
        }
    }
    std::sort(log_files.begin(), log_files.end());

    SnapshotMap result;
    std::regex pattern(R"((\d+),\s*(\d+),\s*(\d+))");

    // 단순히 순차적으로 파싱 (추후 multithreading으로 확장할 수 있음)
    for (const auto &file_path : log_files) {
        // 파일명에서 snapshot index 추출 (예: folio_stats_snapshot_42.log)
        std::string fname = file_path.filename().string();
        int snapshot_idx = 0;
        {
            size_t start = std::string("folio_stats_snapshot_").size();
            size_t end = fname.find(".log");
            std::string idx_str = fname.substr(start, end - start);
            try {
                snapshot_idx = std::stoi(idx_str);
            } catch (...) {
                continue;
            }
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
        result[snapshot_idx] = stats;
    }
    return result;
}

PYBIND11_MODULE(log_parser, m) {
    m.doc() = "Module for fast snapshot log parsing using C++ and multithreading";
    m.def("parse_logs", &parse_logs,
          "Parse snapshot log files from a given directory",
          py::arg("log_dir"), py::arg("num_threads") = 4);
}
