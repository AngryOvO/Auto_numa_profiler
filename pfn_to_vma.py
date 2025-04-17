#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
import ctypes

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
            "Initializes folio_stats using syscall 464, executes a workload, sends "
            "its PID via syscall 463, and collects data from "
            "/sys/kernel/debug/numa_folio/folio_stats every interval seconds. "
            "Each snapshot is saved as a separate log file."
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
    args = parser.parse_args()

    if not args.command:
        print("Error: No workload command provided.")
        sys.exit(1)

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

    print("Starting data collection. Each snapshot will be saved as a separate log file...")
    snapshot = 0
    try:
        while proc.poll() is None:
            snapshot += 1
            try:
                with open("/sys/kernel/debug/numa_folio/folio_stats", "r") as f:
                    data = f.read()
            except Exception as e:
                print(f"Error reading /sys/kernel/debug/numa_folio/folio_stats: {e}")
                break

            # 타임스탬프 없이 snapshot 번호만 활용하여 파일 이름 지정
            filename = f"folio_stats_snapshot_{snapshot}.log"
            with open(filename, "w") as log_file:
                log_file.write(f"Snapshot {snapshot}\n")
                log_file.write("=" * 80 + "\n")
                log_file.write(data)

            print(f"Snapshot {snapshot} saved to '{filename}'.")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Data collection interrupted by user. Terminating workload...")
        proc.terminate()
        proc.wait()

    print("Workload completed and data collection finished.")

if __name__ == "__main__":
    main()
