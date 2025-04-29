// log_parser.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>

namespace py = pybind11;
namespace fs = std::filesystem;

// 각 레코드는 (pfn, refcount) 2개의 int 값을 가짐
using Record = std::tuple<int, int>;
using SnapshotMap = std::unordered_map<int, std::vector<Record>>;

// 직접 문자열 파싱: "pfn,refcount" 형식
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

py::dict parse_logs_numpy(const std::string &log_dir, int num_threads) {
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

    // 전역 결과 맵과 동기화를 위한 변수들
    SnapshotMap global_result;
    std::mutex result_mutex;
    std::atomic<size_t> file_index(0);
    const size_t total_files = log_files.size();

    // worker 스레드는 원자적 인덱스를 통해 남은 파일을 동적으로 가져갑니다.
    auto worker = [&]() {
        SnapshotMap local_result;
        while (true) {
            size_t i = file_index.fetch_add(1);
            if (i >= total_files)
                break;
            const fs::path &file_path = log_files[i];

            // 파일명에서 snapshot index 추출 (예: "folio_stats_snapshot_42.log")
            std::string fname = file_path.filename().string();
            int snapshot_idx = 0;
            try {
                size_t prefix_len = std::string("folio_stats_snapshot_").size();
                size_t pos = fname.find(".log", prefix_len);
                std::string num_str = fname.substr(prefix_len, pos - prefix_len);
                snapshot_idx = std::stoi(num_str);
            } catch (...) {
                continue;
            }

            std::ifstream infile(file_path);
            if (!infile.is_open())
                continue;
            std::vector<Record> records;
            std::string line;
            while (std::getline(infile, line)) {
                if (line.empty())
                    continue;
                int pfn, refcount;
                if (parse_line(line, pfn, refcount))
                    records.push_back(std::make_tuple(pfn, refcount));
            }
            local_result[snapshot_idx] = std::move(records);
        }
        // 로컬 결과를 전역 결과에 병합. 동일 인덱스가 있으면 벡터를 합칩니다.
        if (!local_result.empty()) {
            std::lock_guard<std::mutex> lock(result_mutex);
            for (auto &pair : local_result) {
                auto it = global_result.find(pair.first);
                if (it != global_result.end()) {
                    it->second.insert(it->second.end(),
                        std::make_move_iterator(pair.second.begin()),
                        std::make_move_iterator(pair.second.end()));
                } else {
                    global_result.insert(std::move(pair));
                }
            }
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

    // 결과를 파이썬 dict로 변환하면서, 각 스냅샷의 데이터는 NumPy 배열로 생성합니다.
    py::dict py_result;
    for (const auto &item : global_result) {
        int snapshot_idx = item.first;
        const auto &records = item.second;
        ssize_t num_records = records.size();
        // NumPy 배열의 shape를 [num_records, 2]로 생성: (pfn, refcount)
        std::vector<py::ssize_t> shape = { static_cast<py::ssize_t>(num_records), 2 };
        py::array_t<int> arr(shape);
        auto r = arr.mutable_unchecked<2>();  // 빠른 인덱싱을 위한 unchecked 접근자
        for (ssize_t i = 0; i < num_records; i++) {
            r(i, 0) = std::get<0>(records[i]);
            r(i, 1) = std::get<1>(records[i]);
        }
        py_result[py::int_(snapshot_idx)] = arr;
    }
    return py_result;
}

PYBIND11_MODULE(log_parser, m) {
    m.doc() = "Optimized log parser returning numpy arrays directly. Each numpy array has shape (num_records, 2) representing (pfn, refcount).";
    m.def("parse_logs_numpy", &parse_logs_numpy,
          "Parse snapshot log files and return a dict mapping snapshot index to numpy arrays.\n"
          "Each numpy array has shape (num_records, 2) representing (pfn, refcount).",
          py::arg("log_dir"), py::arg("num_threads") = 4);
}
