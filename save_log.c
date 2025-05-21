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
#include <time.h>
#include <dirent.h>

#define SYS_FOLIO_STAT_RESET         462
#define SYS_SEND_PID                 463
#define SYS_START_GLOBAL_FOLIO_LOG   464
#define SYS_STOP_GLOBAL_FOLIO_LOG    465

#define DEBUGFS_DIR "/sys/kernel/debug/numa_profiler"

/* 이전에 작성한 helper 함수들 (execute_folio_stat_reset, send_pid_to_syscall, create_directory, collect_data 등) 그대로 유지 */

void execute_folio_stat_reset() {
    printf("Executing folio_stat_reset syscall (%d)...\n", SYS_FOLIO_STAT_RESET);
    if (syscall(SYS_FOLIO_STAT_RESET) < 0) {
        perror("Error: folio_stat_reset syscall failed");
        exit(EXIT_FAILURE);
    }
}

void send_pid_to_syscall(pid_t pid) {
    printf("Sending workload PID %d using syscall (%d)...\n", pid, SYS_SEND_PID);
    if (syscall(SYS_SEND_PID, pid) < 0) {
        perror("Error: syscall to send PID failed");
        exit(EXIT_FAILURE);
    }
}

void create_directory(const char *path) {
    struct dirent *entry;
    DIR *dir = opendir(path);

    if (dir) {
        while ((entry = readdir(dir)) != NULL) {
            if (strstr(entry->d_name, ".log")) {
                char filepath[512];
                snprintf(filepath, sizeof(filepath), "%s/%s", path, entry->d_name);
                if (unlink(filepath) < 0)
                    perror("Error deleting old log file");
                else
                    printf("Deleted old log file: %s\n", filepath);
            }
        }
        closedir(dir);
    } else {
        if (mkdir(path, 0755) < 0 && errno != EEXIST) {
            perror("Error creating directory");
            exit(EXIT_FAILURE);
        }
    }
}

/*
 * 기존과 같이 여러 debugfs 파일(노드별 folio_stats, node_pfn_stats 등)을 하나의 스냅샷 파일에 통합해서 저장하는 함수
 */
void collect_data(const char *log_dir, pid_t workload_pid, float interval) {
    char buffer[65536]; // 64KB 버퍼
    int snapshot = 0;

    printf("Collecting data into snapshot files in %s until workload finishes...\n", log_dir);

    while (1) {
        if (waitpid(workload_pid, NULL, WNOHANG) > 0) {
            printf("Workload process has exited. Stopping data collection.\n");
            break;
        }

        char filename[256];
        snprintf(filename, sizeof(filename), "%s/folio_profiler_snapshot_%d.log", log_dir, ++snapshot);
        FILE *output_file = fopen(filename, "w");
        if (!output_file) {
            perror("Error opening snapshot file");
            break;
        }

        DIR *dir = opendir(DEBUGFS_DIR);
        if (!dir) {
            perror("Error opening debugfs directory");
            fclose(output_file);
            break;
        }

        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL) {
            if (strcmp(entry->d_name, ".") == 0 ||
                strcmp(entry->d_name, "..") == 0)
                continue;
            
            if (strstr(entry->d_name, "folio_stats") == NULL)
                continue;

            char file_path[512];
            snprintf(file_path, sizeof(file_path), "%s/%s", DEBUGFS_DIR, entry->d_name);

            int fd = open(file_path, O_RDONLY);
            if (fd < 0) {
                perror("Error opening debugfs file");
                continue;
            }

            fprintf(output_file, "=== File: %s ===\n", entry->d_name);

            ssize_t total_bytes_read = 0;
            while (1) {
                ssize_t bytes_read = read(fd, buffer, sizeof(buffer) - 1);
                if (bytes_read < 0) {
                    perror("Error reading debugfs file");
                    break;
                }
                if (bytes_read == 0)
                    break;

                buffer[bytes_read] = '\0';
                fprintf(output_file, "%s", buffer);
                total_bytes_read += bytes_read;
            }
            fprintf(output_file, "\n\n");
            close(fd);
            printf("Read %zd bytes from %s\n", total_bytes_read, file_path);
        }
        closedir(dir);
        fclose(output_file);
        printf("Snapshot %d saved to %s\n", snapshot, filename);

        usleep((int)(interval * 1000000));
    }
    printf("Data collection completed. %d snapshots saved in %s\n", snapshot, log_dir);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s [--log_dir <log_dir>] [--interval <interval>] [--global] <command> [args...]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *log_dir = "folio_logs";
    float interval = 1.0;
    int global_mode = 0; // global profiling 모드 여부
    char **command = NULL;

    // 명령줄 인자 파싱
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--log_dir") == 0 && i + 1 < argc) {
            log_dir = argv[++i];
        } else if (strcmp(argv[i], "--interval") == 0 && i + 1 < argc) {
            interval = atof(argv[++i]);
        } else if (strcmp(argv[i], "--global") == 0) {
            global_mode = 1;
        } else {
            command = &argv[i];
            break;
        }
    }

    if (!command) {
        fprintf(stderr, "Error: No workload command provided.\n");
        return EXIT_FAILURE;
    }

    create_directory(log_dir);

    /* folio_stat 초기화 syscall 호출 */
    execute_folio_stat_reset();

    /* 워크로드 실행 */
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

    /* 프로파일링 모드에 따라 시스템 콜 호출 
       --global 옵션이 설정되면 전체 시스템 프로파일링, 
       그렇지 않으면 특정 프로세스(PID) 프로파일링 */
    if (global_mode) {
        printf("Starting global folio log profiling...\n");
        if (syscall(SYS_START_GLOBAL_FOLIO_LOG) < 0) {
            perror("Error starting global folio log profiling");
            exit(EXIT_FAILURE);
        }
    } else {
        sleep(1);
        send_pid_to_syscall(pid);
    }

    /* 데이터 수집 */
    collect_data(log_dir, pid, interval);

    waitpid(pid, NULL, 0);

    /* global 모드인 경우 profiling 종료 */
    if (global_mode) {
        if (syscall(SYS_STOP_GLOBAL_FOLIO_LOG) < 0) {
            perror("Error stopping global folio log profiling");
        }
    }

    printf("Workload completed and data collection finished.\n");
    return EXIT_SUCCESS;
}