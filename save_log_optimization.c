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
#include <openssl/sha.h>

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

void sha256_hash_file(const char *filename, unsigned char *output) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Error opening file for hashing");
        exit(EXIT_FAILURE);
    }

    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    unsigned char buffer[4096];
    size_t bytesRead;

    while ((bytesRead = fread(buffer, 1, sizeof(buffer), file)) != 0) {
        SHA256_Update(&sha256, buffer, bytesRead);
    }

    SHA256_Final(output, &sha256);
    fclose(file);
}

int hash_equal(unsigned char *a, unsigned char *b) {
    return memcmp(a, b, SHA256_DIGEST_LENGTH) == 0;
}

void dedup_snapshots(const char *log_dir) {
    struct dirent **namelist;
    int n = scandir(log_dir, &namelist, NULL, alphasort);
    if (n < 0) {
        perror("scandir failed");
        return;
    }

    // 구조체에 파일명과 해시 저장
    typedef struct {
        char filename[256];
        unsigned char hash[SHA256_DIGEST_LENGTH];
        int keep;
    } Snapshot;

    Snapshot *snapshots = calloc(n, sizeof(Snapshot));
    int count = 0;

    for (int i = 0; i < n; i++) {
        if (strstr(namelist[i]->d_name, "folio_stats_snapshot_") && strstr(namelist[i]->d_name, ".log")) {
            snprintf(snapshots[count].filename, sizeof(snapshots[count].filename),
                     "%s/%s", log_dir, namelist[i]->d_name);
            sha256_hash_file(snapshots[count].filename, snapshots[count].hash);
            snapshots[count].keep = 0;
            count++;
        }
        free(namelist[i]);
    }
    free(namelist);

    // 중복 체크
    for (int i = 0; i < count; ) {
        int group_start = i;
        int group_end = i;

        for (int j = i + 1; j < count; j++) {
            if (hash_equal(snapshots[i].hash, snapshots[j].hash)) {
                group_end = j;
            } else {
                break;
            }
        }

        // 동일한 그룹에서 앞의 두 개만 keep
        for (int k = group_start; k <= group_end && k < group_start + 2; k++) {
            snapshots[k].keep = 1;
        }

        i = group_end + 1;
    }

    // 파일 삭제 및 재정렬
    int new_index = 1;
    for (int i = 0; i < count; i++) {
        if (snapshots[i].keep) {
            char new_name[256];
            snprintf(new_name, sizeof(new_name), "%s/folio_stats_snapshot_%d.log", log_dir, new_index++);
            if (rename(snapshots[i].filename, new_name) != 0) {
                perror("rename failed");
            }
        } else {
            unlink(snapshots[i].filename);
        }
    }

    free(snapshots);
    printf("Deduplication complete. Remaining snapshots: %d\n", new_index - 1);
}

void collect_data(const char *log_dir, pid_t workload_pid) {
    char buffer[65536];
    int snapshot = 0;

    printf("Collecting data into snapshot files in %s until workload finishes...\n", log_dir);

    while (1) {
        if (waitpid(workload_pid, NULL, WNOHANG) > 0) {
            printf("Workload process has exited. Stopping data collection.\n");
            break;
        }

        int fd = open("/sys/kernel/debug/numa_folio/folio_stats", O_RDONLY);
        if (fd < 0) {
            perror("Error opening /sys/kernel/debug/numa_folio/folio_stats");
            break;
        }

        char filename[256];
        snprintf(filename, sizeof(filename), "%s/folio_stats_snapshot_%d.log", log_dir, ++snapshot);
        FILE *output_file = fopen(filename, "w");
        if (!output_file) {
            perror("Error opening snapshot file");
            close(fd);
            break;
        }

        ssize_t bytes_read;
        while ((bytes_read = read(fd, buffer, sizeof(buffer) - 1)) > 0) {
            buffer[bytes_read] = '\0';
            fprintf(output_file, "%s", buffer);
        }

        fclose(output_file);
        close(fd);

        printf("Snapshot %d saved to %s\n", snapshot, filename);
        sleep(1);
    }

    printf("Data collection completed. %d snapshots saved in %s\n", snapshot, log_dir);
    dedup_snapshots(log_dir);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s --log_dir <log_dir> --interval <interval> <command> [args...]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *log_dir = "folio_logs";
    float interval = 1.0;
    char **command = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--log_dir") == 0 && i + 1 < argc) {
            log_dir = argv[++i];
        } else if (strcmp(argv[i], "--interval") == 0 && i + 1 < argc) {
            interval = atof(argv[++i]); // 현재 사용하지 않지만 인터페이스 유지
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
    execute_folio_stat_reset();

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

    sleep(1);
    send_pid_to_syscall(pid);
    collect_data(log_dir, pid);

    waitpid(pid, NULL, 0);
    printf("Workload completed and data collection finished.\n");

    return EXIT_SUCCESS;
}
