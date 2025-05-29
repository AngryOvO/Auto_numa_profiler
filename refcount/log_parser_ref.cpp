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
// pfn_stats 파일에서 모든 노드의 PFN 범위를 동적으로 파싱하는 함수
// 파일 포맷 예:
//   node 0
//   start pfn: 1, end pfn: 5767168
//   node 1
//   start pfn: 5767168, end pfn: 11010048
//   node 2
//   start pfn: 11010048, end pfn: 18874368
// 각 노드에 대해 (node, start, end) 튜플을 vector에 담아 반환합니다.
std::vector<std::tuple<int, int, int>> get_all_node_ranges(const std::string &pfn_stats_path) {
    std::ifstream infile(pfn_stats_path);
    if (!infile.is_open())
        throw std::runtime_error("Failed to open pfn_stats file: " + pfn_stats_path);

    std::vector<std::tuple<int, int, int>> ranges;
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        // "node"로 시작하는 라인을 찾음
        if (line.rfind("node", 0) == 0) {
            std::istringstream iss(line);
            std::string dummy;
            int nid;
            iss >> dummy >> nid;  // 예: "node 0" -> nid = 0
            // 다음 줄에서 PFN 범위를 읽음
            if (std::getline(infile, line)) {
                // 예: "start pfn: 1, end pfn: 5767168"
                std::regex range_regex(R"(start\s+pfn:\s*(\d+),\s*end\s+pfn:\s*(\d+))");
                std::smatch match;
                if (std::regex_search(line, match, range_regex)) {
                    if (match.size() == 3) {
                        int start = std::stoi(match[1].str());
                        int end = std::stoi(match[2].str());
                        ranges.push_back(std::make_tuple(nid, start, end));
                    }
                }
            }
        }
    }
    return ranges;
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
// 전체 전역 배열(노드별로 개별 배열)을 미리 할당하는 방식
//  - 각 노드에 대해, PFN 범위에 맞는 NumPy 배열을 할당 (shape = [total_pfns, num_snapshots])
//  - 스냅샷 파일들을 멀티스레드로 파싱하여, 각 로그 파일(스냅샷)에 대해 기록된 pfn, refcount 값을
//    해당하는 노드의 배열에 업데이트합니다.
// 반환값: 노드 번호를 key로, 해당 NumPy 배열을 value로 하는 dict.
py::dict parse_logs_global_array_all_nodes(const std::string &log_dir, int num_threads) {
    // pfn_stats 파일 경로
    std::string pfn_stats_file = "/sys/kernel/debug/numa_profiler/node_pfn_stats";
    // 모든 노드의 PFN 범위 읽기
    auto node_ranges = get_all_node_ranges(pfn_stats_file);
    if (node_ranges.empty())
        throw std::runtime_error("No node ranges found in pfn_stats file.");

    // 정렬: 노드 번호 순으로 정렬
    std::sort(node_ranges.begin(), node_ranges.end(), [](auto &a, auto &b) {
        return std::get<0>(a) < std::get<0>(b);
    });

    // 1. 로그 파일 목록 수집 (파일명이 "folio_stats_snapshot_*.log")
    std::vector<fs::path> log_files;
    for (const auto &entry : fs::directory_iterator(log_dir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            // 필터: 접두사와 확장자 확인
            if (filename.rfind("folio_profiler_snapshot_", 0) == 0 &&
                entry.path().extension() == ".log") {
                log_files.push_back(entry.path());
            }
        }
    }
    // 숫자 순 정렬: 파일명에서 snapshot 번호를 추출해서 비교
    std::regex num_regex("folio_profiler_snapshot_(\\d+)\\.log");
    std::sort(log_files.begin(), log_files.end(), [&](const fs::path &a, const fs::path &b) {
        std::smatch match_a, match_b;
        std::string name_a = a.filename().string();
        std::string name_b = b.filename().string();
        int num_a = 0, num_b = 0;
        if (std::regex_search(name_a, match_a, num_regex) && match_a.size() == 2) {
            num_a = std::stoi(match_a[1].str());
        }
        if (std::regex_search(name_b, match_b, num_regex) && match_b.size() == 2) {
            num_b = std::stoi(match_b[1].str());
        }
        return num_a < num_b;
    });
    size_t num_snapshots = log_files.size();

    // 각 노드별로 NumPy 배열을 할당 및 결과 dict에 저장.
    // 노드 정보 구조체
    struct NodeInfo {
        int node;
        int start;
        size_t total_pfns;
    };
    std::vector<NodeInfo> nodes;
    // 각 배열에 대한 포인터들을 저장 (각 배열은 row-major order)
    std::vector<py::array_t<int>> node_arrays;
    std::vector<int*> data_ptrs;
    py::dict result;
    for (auto &entry : node_ranges) {
        int node = std::get<0>(entry);
        int start = std::get<1>(entry);
        int end = std::get<2>(entry);
        size_t total_pfns = static_cast<size_t>(end - start + 1);
        nodes.push_back({node, start, total_pfns});
        std::vector<py::ssize_t> shape = { static_cast<py::ssize_t>(total_pfns),
                                           static_cast<py::ssize_t>(num_snapshots) };
        py::array_t<int> arr(shape);
        node_arrays.push_back(arr);
        result[py::int_(node)] = arr;
    }
    // 각 배열의 데이터 포인터 추출
    for (auto &arr : node_arrays) {
        auto buf = arr.request();
        int* ptr = static_cast<int*>(buf.ptr);
        data_ptrs.push_back(ptr);
    }

    // 멀티스레드 처리: 각 스냅샷 파일(열)을 처리한다.
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
            while (std::getline(infile, line)) {
                if (line.empty())
                    continue;
                int pfn, refcount;
                if (!parse_line(line, pfn, refcount))
                    continue;
                // 각 노드에 대해, 만약 기록된 pfn가 해당 노드 범위에 속하면 업데이트한다.
                for (size_t ni = 0; ni < nodes.size(); ni++) {
                    const auto &node_info = nodes[ni];
                    if (pfn >= node_info.start && pfn < node_info.start + static_cast<int>(node_info.total_pfns)) {
                        int local_index = pfn - node_info.start;
                        // 배열의 layout이 row-major: offset = local_index * num_snapshots + idx
                        data_ptrs[ni][local_index * num_snapshots + idx] = refcount;
                        break;  // 한 pfn은 단 하나의 노드에 속함.
                    }
                }
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

    return result;
}

PYBIND11_MODULE(log_parser, m) {
    m.doc() = "Global-array based log parser for all nodes. "
              "Reads NUMA node PFN ranges from '/sys/kernel/debug/numa_profiler/node_pfn_stats' and allocates, for each node, "
              "a NumPy array with shape (total_pfns, num_snapshots) (row: node-local PFN, column: snapshot). "
              "Each cell contains the refcount for that PFN at that snapshot.";
    m.def("parse_logs_global_array_all_nodes", &parse_logs_global_array_all_nodes,
          "Parse snapshot log files and update a global NumPy array for each node. "
          "Returns a dict mapping node ID to a NumPy array (shape: (total_pfns, num_snapshots)) "
          "with refcount values updated using node-local indexing.",
          py::arg("log_dir"), py::arg("num_threads") = 4);
}
