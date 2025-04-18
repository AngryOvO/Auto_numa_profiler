#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <sys/stat.h>

#define SYS_FOLIO_STAT_RESET 462
#define SYS_SEND_PID 463

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
    if (mkdir(path, 0755) < 0 && errno != EEXIST) {
        perror("Error creating directory");
        exit(EXIT_FAILURE);
    }
}

void save_node_data(const char *log_dir, int snapshot, int node, const char *data) {
    char node_dir[256];
    char filename[256];
    snprintf(node_dir, sizeof(node_dir), "%s/node_%d", log_dir, node);
    create_directory(node_dir);

    snprintf(filename, sizeof(filename), "%s/folio_stats_snapshot_%d.log", node_dir, snapshot);
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error opening snapshot file");
        exit(EXIT_FAILURE);
    }

    fprintf(file, "%s\n", data);
    fclose(file);
}

void parse_pfn_stats(const char *pfn_stats_path, int *node_start_pfn, int *node_end_pfn, int max_nodes) {
    FILE *file = fopen(pfn_stats_path, "r");
    if (!file) {
        perror("Error opening pfn_stats file");
        exit(EXIT_FAILURE);
    }

    char line[256];
    int current_node = -1;
    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line, "node %d", &current_node) == 1) {
            // 노드 번호를 읽음
            if (current_node < 0 || current_node >= max_nodes) {
                fprintf(stderr, "Invalid node number in pfn_stats: %d\n", current_node);
                continue;
            }
        } else if (current_node >= 0 && sscanf(line, "start pfn: %d, end pfn: %d", &node_start_pfn[current_node], &node_end_pfn[current_node]) == 2) {
            // 현재 노드의 PFN 범위를 읽음
            printf("Node %d: start pfn = %d, end pfn = %d\n", current_node, node_start_pfn[current_node], node_end_pfn[current_node]);
        }
    }

    fclose(file);
}

void collect_data(const char *log_dir, float interval, const char *pfn_stats_path) {
    int snapshot = 0;
    char buffer[65536]; // 더 큰 버퍼로 설정 (64KB)

    // PFN 범위 초기화
    int node_start_pfn[256] = {0};
    int node_end_pfn[256] = {0};
    parse_pfn_stats(pfn_stats_path, node_start_pfn, node_end_pfn, 256);

    while (1) {
        snapshot++;
        int fd = open("/sys/kernel/debug/numa_folio/folio_stats", O_RDONLY);
        if (fd < 0) {
            perror("Error reading /sys/kernel/debug/numa_folio/folio_stats");
            break;
        }

        ssize_t bytes_read = read(fd, buffer, sizeof(buffer) - 1);
        close(fd);
        if (bytes_read < 0) {
            perror("Error reading data");
            break;
        }

        buffer[bytes_read] = '\0'; // null-terminate the buffer

        // 노드별 데이터를 저장할 버퍼 초기화
        char node_data[256][65536] = {0}; // 최대 256개의 노드, 각 노드에 대해 64KB 버퍼
        int node_data_lengths[256] = {0}; // 각 노드의 데이터 길이 추적

        // 데이터를 한 줄씩 파싱
        char *line = strtok(buffer, "\n");
        while (line) {
            int node, pfn, source_nid, migrate_count;
            if (sscanf(line, "%d,%d,%d,%d", &node, &pfn, &source_nid, &migrate_count) == 4) {
                // PFN 범위 확인
                if (node >= 0 && node < 256 && pfn >= node_start_pfn[node] && pfn <= node_end_pfn[node]) {
                    // 노드 데이터에 현재 라인을 추가
                    int len = snprintf(node_data[node] + node_data_lengths[node],
                                       sizeof(node_data[node]) - node_data_lengths[node],
                                       "%s\n", line);
                    if (len > 0) {
                        node_data_lengths[node] += len;
                    }
                } else {
                    fprintf(stderr, "Skipping line outside PFN range: %s\n", line);
                }
            } else {
                fprintf(stderr, "Skipping malformed line: %s\n", line);
            }
            line = strtok(NULL, "\n");
        }

        // 노드별로 스냅샷 파일 생성
        for (int node = 0; node < 256; node++) {
            if (node_data_lengths[node] > 0) {
                save_node_data(log_dir, snapshot, node, node_data[node]);
            }
        }

        sleep((unsigned int)interval);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s --log_dir <log_dir> --interval <interval> --pfn_stats <pfn_stats_path> <command> [args...]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *log_dir = "folio_logs";
    float interval = 1.0;
    const char *pfn_stats_path = "/sys/kernel/debug/numa_folio/pfn_stats";
    char **command = NULL;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--log_dir") == 0 && i + 1 < argc) {
            log_dir = argv[++i];
        } else if (strcmp(argv[i], "--interval") == 0 && i + 1 < argc) {
            interval = atof(argv[++i]);
        } else if (strcmp(argv[i], "--pfn_stats") == 0 && i + 1 < argc) {
            pfn_stats_path = argv[++i];
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

    // Execute folio_stat_reset syscall
    execute_folio_stat_reset();

    // Execute workload
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

    // Send PID to syscall
    sleep(1);
    send_pid_to_syscall(pid);

    // Collect data
    collect_data(log_dir, interval, pfn_stats_path);

    // Wait for workload to finish
    waitpid(pid, NULL, 0);
    printf("Workload completed and data collection finished.\n");

    return EXIT_SUCCESS;
}