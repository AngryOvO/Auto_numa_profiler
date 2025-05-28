#include <pthread.h>
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

/* 기존 helper 함수들 */

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

/* 멀티스레딩을 위한 thread argument 구조체 */
typedef struct {
    char file_path[512];       // debugfs 파일 전체 경로
    char file_name[256];       // 파일 이름 (헤더에 사용)
    char temp_file_path[512];  // 각 스레드가 쓸 임시 파일 이름
} thread_arg_t;

/* 각 스레드가 실행할 함수: 해당 debugfs 파일을 읽어 임시 파일에 기록 */
void *read_debugfs_file(void *arg) {
    thread_arg_t *targ = (thread_arg_t *)arg;
    int fd = open(targ->file_path, O_RDONLY);
    if (fd < 0) {
        perror("Error opening debugfs file in thread");
        pthread_exit(NULL);
    }
    
    FILE *temp_fp = fopen(targ->temp_file_path, "w");
    if (!temp_fp) {
        perror("Error opening temporary file in thread");
        close(fd);
        pthread_exit(NULL);
    }
    
    // 헤더 작성
    fprintf(temp_fp, "=== File: %s ===\n", targ->file_name);
    
    char buffer[65536];
    ssize_t bytes_read;
    while ((bytes_read = read(fd, buffer, sizeof(buffer) - 1)) > 0) {
        buffer[bytes_read] = '\0';
        fputs(buffer, temp_fp);
    }
    if (bytes_read < 0) {
        perror("Error reading debugfs file in thread");
    }
    
    fprintf(temp_fp, "\n\n");
    
    fclose(temp_fp);
    close(fd);
    pthread_exit(NULL);
}

/*
 * 멀티스레드를 활용하여 각 debugfs 파일(노드별 folio_stats)을 읽고 각 스레드가 임시 파일에 기록,
 * 이후 이 임시 파일들을 하나의 스냅샷 파일로 병합하는 함수
 */
void collect_data_multithread(const char *log_dir, pid_t workload_pid, float interval) {
    int snapshot = 0;
    printf("Collecting data into snapshot files in %s until workload finishes...\n", log_dir);

    while (1) {
        if (waitpid(workload_pid, NULL, WNOHANG) > 0) {
            printf("Workload process has exited. Stopping data collection.\n");
            break;
        }
        
        snapshot++; // 새로운 스냅샷 번호
        DIR *dir = opendir(DEBUGFS_DIR);
        if (!dir) {
            perror("Error opening debugfs directory");
            break;
        }
        
        #define MAX_FILES 128
        char file_paths[MAX_FILES][512];
        char file_names[MAX_FILES][256];
        int file_count = 0;
        struct dirent *entry;
        while ((entry = readdir(dir)) != NULL && file_count < MAX_FILES) {
            if (strcmp(entry->d_name, ".") == 0 ||
                strcmp(entry->d_name, "..") == 0)
                continue;
            if (strstr(entry->d_name, "folio_stats") == NULL)
                continue;
            snprintf(file_paths[file_count], sizeof(file_paths[file_count]), "%s/%s", DEBUGFS_DIR, entry->d_name);
            snprintf(file_names[file_count], sizeof(file_names[file_count]), "%s", entry->d_name);
            file_count++;
        }
        closedir(dir);

        // 각 파일마다 스레드 생성
        pthread_t threads[MAX_FILES];
        thread_arg_t args[MAX_FILES];
        for (int i = 0; i < file_count; i++) {
            strncpy(args[i].file_path, file_paths[i], sizeof(args[i].file_path));
            strncpy(args[i].file_name, file_names[i], sizeof(args[i].file_name));
            // 임시 파일 이름 예: "<log_dir>/temp_snapshot_<snapshot>_file_<i>.tmp"
            snprintf(args[i].temp_file_path, sizeof(args[i].temp_file_path),
                     "%s/temp_snapshot_%d_file_%d.tmp", log_dir, snapshot, i);

            if (pthread_create(&threads[i], NULL, read_debugfs_file, &args[i]) != 0) {
                perror("Error creating thread");
            }
        }

        // 모든 스레드가 작업을 마칠 때까지 대기
        for (int i = 0; i < file_count; i++) {
            pthread_join(threads[i], NULL);
        }

        // 임시 파일들을 최종 스냅샷 파일로 병합
        char snapshot_filename[256];
        snprintf(snapshot_filename, sizeof(snapshot_filename), "%s/folio_profiler_snapshot_%d.log", log_dir, snapshot);
        FILE *snapshot_fp = fopen(snapshot_filename, "w");
        if (!snapshot_fp) {
            perror("Error creating snapshot file");
            break;
        }

        for (int i = 0; i < file_count; i++) {
            FILE *temp_fp = fopen(args[i].temp_file_path, "r");
            if (temp_fp) {
                char buffer[4096];
                size_t n;
                while ((n = fread(buffer, 1, sizeof(buffer), temp_fp)) > 0) {
                    fwrite(buffer, 1, n, snapshot_fp);
                }
                fclose(temp_fp);
                if (unlink(args[i].temp_file_path) < 0)
                    perror("Error deleting temporary file");
            }
        }
        fclose(snapshot_fp);
        printf("Snapshot %d saved to %s\n", snapshot, snapshot_filename);

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
        if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--log_dir <log_dir>] [--interval <interval>] [--global] <command> [args...]\n", argv[0]);
            printf("\nOptions:\n");
            printf("  --log_dir <log_dir>  Specify directory to store log files (default: folio_logs)\n");
            printf("  --interval <interval> Set the data collection interval in seconds (default: 1.0)\n");
            printf("  --global             Enable global profiling mode\n");
            printf("  <command> [args...]  Specify the workload command to execute\n");
            exit(EXIT_SUCCESS);
        } else if (strcmp(argv[i], "--log_dir") == 0 && i + 1 < argc) {
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
        // PID 전달 전에 조금 대기해서 워크로드가 안정적으로 시작되도록 함
        sleep(1);
        send_pid_to_syscall(pid);
    }

    /* 멀티스레딩을 적용한 데이터 수집 */
    collect_data_multithread(log_dir, pid, interval);

    waitpid(pid, NULL, 0);

    /* global 모드일 경우 profiling 종료 */
    if (global_mode) {
        if (syscall(SYS_STOP_GLOBAL_FOLIO_LOG) < 0) {
            perror("Error stopping global folio log profiling");
        }
    }

    printf("Workload completed and data collection finished.\n");
    return EXIT_SUCCESS;
}