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

void collect_data(const char *log_dir, float interval, pid_t workload_pid) {
    int snapshot = 0;
    char buffer[65536]; // 64KB 버퍼

    while (1) {
        // 워크로드 프로세스 상태 확인
        if (waitpid(workload_pid, NULL, WNOHANG) > 0) {
            printf("Workload process has exited. Stopping data collection.\n");
            break;
        }

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

        // 스냅샷 파일 생성
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/folio_stats_snapshot_%d.log", log_dir, snapshot);
        FILE *file = fopen(filename, "w");
        if (!file) {
            perror("Error opening snapshot file");
            break;
        }

        fprintf(file, "Snapshot %d\n", snapshot);
        fprintf(file, "================================================================================\n");
        fprintf(file, "%s", buffer);
        fclose(file);

        printf("Snapshot %d saved to %s\n", snapshot, filename);

        sleep((unsigned int)interval);
    }
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

    // 로그 디렉토리 생성
    create_directory(log_dir);

    // folio_stat 초기화 syscall 호출
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

    // PID를 syscall로 전달
    sleep(1);
    send_pid_to_syscall(pid);

    // 데이터 수집
    collect_data(log_dir, interval, pid);

    // 워크로드 종료 대기
    waitpid(pid, NULL, 0);
    printf("Workload completed and data collection finished.\n");

    return EXIT_SUCCESS;
}