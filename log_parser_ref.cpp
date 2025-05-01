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
#include <utility>

namespace py = pybind11;
namespace fs = std::filesystem;

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

py::dict parse_logs_numpy(const std::string &log_dir, int num_threads) {
    // 대상 로그 파일 목록 수집 (파일명이 "folio_stats_snapshot_*.log")
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
    size_t total_files = log_files.size();
    std::atomic<size_t> file_index(0);

    // 각 파일에 대해 Snapshot 정보를 (snapshot_idx, numpy array) 쌍으로 저장할 벡터
    std::vector<std::pair<int, py::array_t<int>>> results;
    std::mutex results_mutex;

    auto worker = [&]() {
        while (true) {
            size_t i = file_index.fetch_add(1);
            if (i >= total_files)
                break;
            const fs::path &file_path = log_files[i];

            // 파일명으로부터 snapshot index 추출
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

            // [첫 번째 패스] : 파일을 한 줄씩 읽어 유효한(비어있지 않고 파싱 가능한) 라인 수를 센다.
            size_t count = 0;
            std::string line;
            while (std::getline(infile, line)) {
                if (line.empty())
                    continue;
                int p, r;
                if (parse_line(line, p, r))
                    count++;
            }

            // NumPy 배열 할당 (shape: [count, 2])
            std::vector<py::ssize_t> shape = { static_cast<py::ssize_t>(count), 2 };
            py::array_t<int> arr(shape);
            auto arr_mut = arr.mutable_unchecked<2>();

            // [두 번째 패스] : 파일의 처음으로 되돌린 후 데이터를 NumPy 배열에 채운다.
            infile.clear();
            infile.seekg(0);
            size_t idx = 0;
            while (std::getline(infile, line)) {
                if (line.empty())
                    continue;
                int p, r;
                if (parse_line(line, p, r)) {
                    arr_mut(idx, 0) = p;
                    arr_mut(idx, 1) = r;
                    idx++;
                }
            }
            infile.close();

            // thread 안전하게 결과 벡터에 추가
            {
                std::lock_guard<std::mutex> lock(results_mutex);
                results.push_back({ snapshot_idx, arr });
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

    // 최종적으로 모든 결과를 Python dict로 조합.
    py::dict py_result;
    for (auto &pair : results) {
        py_result[py::int_(pair.first)] = pair.second;
    }
    return py_result;
}

PYBIND11_MODULE(log_parser, m) {
    m.doc() = "Memory-efficient log parser returning numpy arrays. Each numpy array has shape (num_records, 2) representing (pfn, refcount).";
    m.def("parse_logs_numpy", &parse_logs_numpy,
          "Parse snapshot log files and return a dict mapping snapshot index to numpy arrays.\n"
          "Each numpy array has shape (num_records, 2) representing (pfn, refcount).",
          py::arg("log_dir"), py::arg("num_threads") = 4);
}
