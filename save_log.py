#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
import ctypes
import os

# syscall 462를 호출하여 folio_stat을 초기화하는 함수
def execute_folio_stat_reset():
    SYS_FOLIO_STAT_RESET = 462  # folio_stat 초기화 syscall 번호
    libc = ctypes.CDLL("libc.so.6")
    print("Executing folio_stat_reset syscall (464)...")
    ret = libc.syscall(SYS_FOLIO_STAT_RESET)
    if ret < 0:
        print(f"Error: folio_stat_reset syscall (464) failed with return code {ret}.")
        sys.exit(1)

# syscall 463를 호출하여 워크로드의 pid를 전달하는 함수
def send_pid_to_syscall(pid):
    SYS_SEND_PID = 463  # pid 전달용 syscall 번호
    libc = ctypes.CDLL("libc.so.6")
    print(f"Sending workload PID {pid} using syscall 463...")
    ret = libc.syscall(SYS_SEND_PID, pid)
    if ret < 0:
        print(f"Error: syscall 463 with pid {pid} failed with return code {ret}.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Initializes folio_stats using syscall 462, executes a workload, sends "
            "its PID via syscall 463, and collects data from "
            "/sys/kernel/debug/numa_folio/folio_stats every interval seconds. "
            "Each snapshot is saved as a separate log file in a designated directory."
        )
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Workload command to execute (e.g., '/bin/sleep 10')."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Data collection interval in seconds (default: 1.0 sec)."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="folio_logs",
        help="Directory where snapshot log files will be saved (default: folio_logs)."
    )
    args = parser.parse_args()

    if not args.command:
        print("Error: No workload command provided.")
        sys.exit(1)

    # 지정된 로그 디렉터리 생성 (존재하지 않으면)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # folio_stat 초기화를 위한 syscall 464 실행
    execute_folio_stat_reset()

    # 워크로드 실행
    print("Executing workload:", " ".join(args.command))
    try:
        proc = subprocess.Popen(args.command)
    except FileNotFoundError:
        print(f"Error: Command '{args.command[0]}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error executing workload command '{args.command}': {e}")
        sys.exit(1)

    # 워크로드 실행 후 잠깐 대기한 뒤 pid를 syscall 463로 전달
    time.sleep(0.1)
    send_pid_to_syscall(proc.pid)

    print("Starting data collection. Each snapshot will be saved as separate log files for each node in '{}'...".format(args.log_dir))
    snapshot = 0
    try:
        while proc.poll() is None:
            snapshot += 1
            try:
                with open("/sys/kernel/debug/numa_folio/folio_stats", "r") as f:
                    data = f.readlines()  # 각 줄을 읽어 리스트로 저장
            except Exception as e:
                print(f"Error reading /sys/kernel/debug/numa_folio/folio_stats: {e}")
                break

            # 노드별 데이터를 저장할 딕셔너리 초기화
            node_data = {}

            # 데이터를 노드별로 분리
            for line in data:
                try:
                    node, pfn, source_nid, migrate_count = map(int, line.strip().split(","))
                    if node not in node_data:
                        node_data[node] = []
                    node_data[node].append(line.strip())
                except ValueError:
                    print(f"Skipping malformed line: {line.strip()}")
                    continue

            # 노드별로 스냅샷 파일 생성
            for node, lines in node_data.items():
                node_dir = os.path.join(args.log_dir, f"node_{node}")
                if not os.path.exists(node_dir):
                    os.makedirs(node_dir)

                filename = os.path.join(node_dir, f"folio_stats_snapshot_{snapshot}.log")
                with open(filename, "w") as log_file:
                    log_file.write("\n".join(lines) + "\n")

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Data collection interrupted by user. Terminating workload...")
        proc.terminate()
        proc.wait()

    print("Workload completed and data collection finished.")

if __name__ == "__main__":
    main()
