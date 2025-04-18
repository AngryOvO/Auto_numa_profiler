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

void collect_data(const char *log_dir, float interval) {
    int snapshot = 0;
    char buffer[1024];
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

        buffer[bytes_read] = '\0';

        // Parse data and save by node
        char *line = strtok(buffer, "\n");
        while (line) {
            int node, pfn, source_nid, migrate_count;
            if (sscanf(line, "%d,%d,%d,%d", &node, &pfn, &source_nid, &migrate_count) == 4) {
                save_node_data(log_dir, snapshot, node, line);
            } else {
                fprintf(stderr, "Skipping malformed line: %s\n", line);
            }
            line = strtok(NULL, "\n");
        }

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

    // Parse arguments
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
    collect_data(log_dir, interval);

    // Wait for workload to finish
    waitpid(pid, NULL, 0);
    printf("Workload completed and data collection finished.\n");

    return EXIT_SUCCESS;
}