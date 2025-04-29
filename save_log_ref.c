#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#include <dirent.h>
#include <time.h>

#define SYS_FOLIO_STAT_RESET 462
#define SYS_SEND_PID 463

// folio_stat_reset syscall을 호출하여 참조 카운트 통계 초기화를 수행합니다.
void execute_folio_stat_reset() {
    printf("Executing folio_stat_reset syscall (%d) for reference count reset...\n", SYS_FOLIO_STAT_RESET);
    if (syscall(SYS_FOLIO_STAT_RESET) < 0) {
        perror("Error: folio_stat_reset syscall failed");
        exit(EXIT_FAILURE);
    }
}

// 워크로드 PID를 커널로 전송합니다.
void send_pid_to_syscall(pid_t pid) {
    printf("Sending workload PID %d using syscall (%d)...\n", pid, SYS_SEND_PID);
    if (syscall(SYS_SEND_PID, pid) < 0) {
        perror("Error: syscall to send PID failed");
        exit(EXIT_FAILURE);
    }
}

// 주어진 경로에 있는 디렉토리를 생성하거나, 이미 존재하면 이전의 .log 파일들을 삭제합니다.
void create_directory(const char *path) {
    struct dirent *entry;
    DIR *dir = opendir(path);

    if (dir) {
        // 디렉토리가 존재하면, 이전 스냅샷 .log 파일들을 삭제합니다.
        while ((entry = readdir(dir)) != NULL) {
            if (strstr(entry->d_name, ".log")) {
                char filepath[512];
                snprintf(filepath, sizeof(filepath), "%s/%s", path, entry->d_name);
                if (unlink(filepath) < 0) {
                    perror("Error deleting old log file");
                } else {
                    printf("Deleted old log file: %s\n", filepath);
                }
            }
        }
        closedir(dir);
    } else {
        // 디렉토리가 없으면 새로 생성합니다.
        if (mkdir(path, 0755) < 0 && errno != EEXIST) {
            perror("Error creating directory");
            exit(EXIT_FAILURE);
        }
    }
}

// debugfs의 folio_stats 파일은 이제 "pfn,refcount" 형식의 데이터를 출력합니다.
// 이 함수는 워크로드가 실행되는 동안 주기적으로 해당 파일을 읽어 스냅샷 파일에 저장합니다.
void collect_data(const char *log_dir, pid_t workload_pid) {
    char buffer[65536]; // 64KB 버퍼
    int snapshot = 0;

    printf("Collecting reference count data into snapshot files in %s until workload finishes...\n", log_dir);

    while (1) {
        // 워크로드 프로세스 상태 확인
        if (waitpid(workload_pid, NULL, WNOHANG) > 0) {
            printf("Workload process has exited. Stopping data collection.\n");
            break;
        }

        // folio_stats 파일 열기 (이제 각 줄은 "pfn,refcount" 형식입니다.)
        int fd = open("/sys/kernel/debug/numa_folio/folio_stats", O_RDONLY);
        if (fd < 0) {
            perror("Error opening /sys/kernel/debug/numa_folio/folio_stats");
            break;
        }

        ssize_t total_bytes_read = 0;
        FILE *output_file = NULL;

        // 스냅샷 파일 생성
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/folio_stats_snapshot_%d.log", log_dir, ++snapshot);
        output_file = fopen(filename, "w");
        if (!output_file) {
            perror("Error opening snapshot file");
            close(fd);
            break;
        }

        // folio_stats의 데이터를 읽어 스냅샷 파일에 기록
        while (1) {
            ssize_t bytes_read = read(fd, buffer, sizeof(buffer) - 1);
            if (bytes_read < 0) {
                perror("Error reading /sys/kernel/debug/numa_folio/folio_stats");
                fclose(output_file);
                close(fd);
                return;
            }
            if (bytes_read == 0) {
                // 파일 끝에 도달
                break;
            }
            buffer[bytes_read] = '\0'; // null-terminate the buffer
            fprintf(output_file, "%s", buffer); // 데이터를 파일에 기록
            total_bytes_read += bytes_read;
        }

        fclose(output_file);
        close(fd);

        printf("Snapshot %d saved to %s (%zd bytes read)\n", snapshot, filename, total_bytes_read);

        // 1초 대기 (필요에 따라 interval 값을 조정할 수 있음)
        sleep(1);
    }

    printf("Data collection completed. %d snapshots saved in %s\n", snapshot, log_dir);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s --log_dir <log_dir> --interval <interval> <command> [args...]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *log_dir = "folio_logs";
    float interval = 1.0;
    char **command = NULL;

    // 명령줄 인자 파싱
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--log_dir") == 0 && i + 1 < argc) {
            log_dir = argv[++i];
        } else if (strcmp(argv[i], "--interval") == 0 && i + 1 < argc) {
            interval = atof(argv[++i]);
        } else {
            command = &argv[i];
            break;
        }
    }

    if (!command) {
        fprintf(stderr, "Error: No workload command provided.\n");
        return EXIT_FAILURE;
    }

    // 로그 디렉토리 생성 또는 초기화
    create_directory(log_dir);

    // folio_stat_reset syscall 호출 (이제 참조 카운트 통계를 초기화합니다.)
    execute_folio_stat_reset();

    // 워크로드 실행
    printf("Executing workload: %s\n", command[0]);
    pid_t pid = fork();
    if (pid < 0) {
        perror("Error forking process");
        return EXIT_FAILURE;
    } else if (pid == 0) {
        execvp(command[0], command);
        perror("Error executing workload command");
        return EXIT_FAILURE;
    }

    // PID를 커널로 전송
    sleep(1);
    send_pid_to_syscall(pid);

    // 데이터 수집 시작
    collect_data(log_dir, pid);

    // 워크로드 종료 대기
    waitpid(pid, NULL, 0);
    printf("Workload completed and data collection finished.\n");

    return EXIT_SUCCESS;
}
