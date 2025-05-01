// log_parser_ref.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <sstream>
#include <regex>
#include <stdexcept>

namespace py = pybind11;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------
// pfn_stats 파일에서 특정 노드(nid)의 PFN 범위를 동적으로 파싱하는 함수
// 파일 포맷 예:
//   node 0
//   start pfn: 1, end pfn: 5767168
//   node 1
//   start pfn: 5767168, end pfn: 11010048
//   node 2
//   start pfn: 11010048, end pfn: 18874368
// 지정한 노드번호에 대해 (start, end) 쌍을 반환합니다.
std::pair<int, int> get_node_pfn_range(const std::string &pfn_stats_path, int node_id) {
    std::ifstream infile(pfn_stats_path);
    if (!infile.is_open())
        throw std::runtime_error("Failed to open pfn_stats file: " + pfn_stats_path);

    std::string line;
    while (std::getline(infile, line)) {
        // Trim 공백 제거 (필요시)
        if (line.empty()) continue;
        // "node "로 시작하는 라인을 찾음
        if (line.rfind("node", 0) == 0) {
            std::istringstream iss(line);
            std::string dummy;
            int nid;
            iss >> dummy >> nid;  // 예: "node 0" -> dummy="node", nid=0
            if (nid == node_id) {
                // 다음 라인에서 PFN 범위를 읽음
                if (std::getline(infile, line)) {
                    // 예: "start pfn: 1, end pfn: 5767168"
                    std::regex range_regex(R"(start\s+pfn:\s*(\d+),\s*end\s+pfn:\s*(\d+))");
                    std::smatch match;
                    if (std::regex_search(line, match, range_regex)) {
                        if (match.size() == 3) {
                            int start = std::stoi(match[1].str());
                            int end = std::stoi(match[2].str());
                            return std::make_pair(start, end);
                        }
                    }
                    throw std::runtime_error("Failed to parse PFN range on line: " + line);
                } else {
                    throw std::runtime_error("pfn_stats file ended unexpectedly after node line.");
                }
            }
        }
    }
    throw std::runtime_error("Node " + std::to_string(node_id) +
                             " not found in pfn_stats file: " + pfn_stats_path);
}

// ---------------------------------------------------------------------
// 파싱 함수: "pfn,refcount" 형식의 한 줄을 정수 두 개로 변환
bool parse_line(const std::string &line, int &pfn, int &refcount) {
    size_t pos = line.find(',');
    if (pos == std::string::npos)
        return false;
    try {
        pfn = std::stoi(line.substr(0, pos));
        refcount = std::stoi(line.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------
// 전체 전역 배열을 미리 할당하는 방식 (노드별 PFN 범위 기반)
//  - 행: 전체 PFN 개수 (total_pfns)
//  - 열: 스냅샷 수 (로그 파일 개수)
// pfn_stats 파일에서 동적으로 노드 PFN 범위를 얻은 후, 전역 배열에 각 스냅샷별 refcount 값을 업데이트합니다.
py::array_t<int> parse_logs_global_array(const std::string &log_dir, int num_threads) {
    // /sys/kernel/debug/numa_folio/pfn_stats 파일에서 노드 0의 PFN 범위를 동적으로 파싱
    std::string pfn_stats_file = "/sys/kernel/debug/numa_folio/pfn_stats";
    auto pf_range = get_node_pfn_range(pfn_stats_file, 0);
    int node_start_pfn = pf_range.first;
    int node_end_pfn = pf_range.second;
    size_t total_pfns = static_cast<size_t>(node_end_pfn - node_start_pfn + 1);

    // 1. 로그 파일 목록 수집 (파일명이 "folio_stats_snapshot_*.log")
    std::vector<fs::path> log_files;
    for (const auto &entry : fs::directory_iterator(log_dir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.rfind("folio_stats_snapshot_", 0) == 0 &&
                entry.path().extension() == ".log") {
                log_files.push_back(entry.path());
            }
        }
    }
    std::sort(log_files.begin(), log_files.end());

    size_t num_snapshots = log_files.size();
    // 전역 배열 할당: shape = [total_pfns, num_snapshots]
    std::vector<py::ssize_t> shape = { static_cast<py::ssize_t>(total_pfns),
                                       static_cast<py::ssize_t>(num_snapshots) };
    // 초기값 0으로 설정
    py::array_t<int> global_array(shape);
    auto global_mut = global_array.mutable_unchecked<2>();

    // 스냅샷(열) 처리를 위한 thread-safe atomic index
    std::atomic<size_t> snapshot_index(0);

    auto worker = [&]() {
        while (true) {
            size_t idx = snapshot_index.fetch_add(1);
            if (idx >= num_snapshots)
                break;
            fs::path file_path = log_files[idx];

            std::ifstream infile(file_path);
            if (!infile.is_open())
                continue;

            std::string line;
            // 각 로그 파일의 각 라인은 "pfn,refcount" 형태
            while (std::getline(infile, line)) {
                if (line.empty())
                    continue;
                int pfn, refcount;
                if (!parse_line(line, pfn, refcount))
                    continue;
                // 전역 PFN 값을 노드 내 인덱스로 변환:
                // local_index = pfn - node_start_pfn
                int local_index = pfn - node_start_pfn;
                if (local_index < 0 || static_cast<size_t>(local_index) >= total_pfns)
                    continue;
                // 글로벌 배열의 해당 셀에 refcount를 업데이트
                global_mut(local_index, idx) = refcount;
            }
            infile.close();
        }
    };

    if (num_threads < 1)
        num_threads = 1;
    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back(worker);
    }
    for (auto &th : threads)
        th.join();

    return global_array;
}

PYBIND11_MODULE(log_parser, m) {
    m.doc() = "Global-array based log parser. Dynamically reads NUMA node PFN range from '/sys/kernel/debug/numa_folio/pfn_stats' "
              "and allocates one big NumPy array with shape (total_pfns, num_snapshots). Each cell contains the refcount for that PFN "
              "at that snapshot, with indexing adjusted using the node's base PFN.";
    m.def("parse_logs_global_array", &parse_logs_global_array,
          "Parse snapshot log files and update a global NumPy array with shape (total_pfns, num_snapshots).\n"
          "Each cell contains the refcount for that PFN at that snapshot. The PFN values are converted into node-local indices "
          "using the PFN range information from /sys/kernel/debug/numa_folio/pfn_stats.",
          py::arg("log_dir"), py::arg("num_threads") = 4);
}
