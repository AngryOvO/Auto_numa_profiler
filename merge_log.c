#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>

void integrate_snapshots(const char *log_dir, const char *output_file) {
    DIR *dir = opendir(log_dir);
    if (!dir) {
        perror("Error opening log directory");
        exit(EXIT_FAILURE);
    }

    FILE *output = fopen(output_file, "w");
    if (!output) {
        perror("Error opening output file");
        closedir(dir);
        exit(EXIT_FAILURE);
    }

    fprintf(output, "snapshot,node,pfn,source_nid,migrate_count\n"); // CSV 헤더

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, "folio_stats_snapshot_") && strstr(entry->d_name, ".log")) {
            char filepath[256];
            snprintf(filepath, sizeof(filepath), "%s/%s", log_dir, entry->d_name);

            FILE *file = fopen(filepath, "r");
            if (!file) {
                perror("Error opening snapshot file");
                continue;
            }

            char line[256];
            int snapshot = 0;

            // 스냅샷 번호 추출
            sscanf(entry->d_name, "folio_stats_snapshot_%d.log", &snapshot);

            // 파일 내용 읽기
            while (fgets(line, sizeof(line), file)) {
                int node, pfn, source_nid, migrate_count;
                if (sscanf(line, "%d,%d,%d,%d", &node, &pfn, &source_nid, &migrate_count) == 4) {
                    fprintf(output, "%d,%d,%d,%d,%d\n", snapshot, node, pfn, source_nid, migrate_count);
                }
            }

            fclose(file);
        }
    }

    fclose(output);
    closedir(dir);

    printf("Integrated data saved to %s\n", output_file);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <log_dir> <output_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *log_dir = argv[1];
    const char *output_file = argv[2];

    integrate_snapshots(log_dir, output_file);

    return EXIT_SUCCESS;
}