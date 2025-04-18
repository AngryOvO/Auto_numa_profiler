#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>

#define LOG_DIR "folio_logs"          // 고정된 로그 디렉토리 경로
#define OUTPUT_FILE "integrated_data.csv" // 고정된 출력 CSV 파일 이름

void integrate_snapshots() {
    DIR *dir = opendir(LOG_DIR);
    if (!dir) {
        perror("Error opening log directory");
        exit(EXIT_FAILURE);
    }

    FILE *output = fopen(OUTPUT_FILE, "w");
    if (!output) {
        perror("Error opening output file");
        closedir(dir);
        exit(EXIT_FAILURE);
    }

    fprintf(output, "snapshot,node,pfn,source_nid,migrate_count\n"); // CSV 헤더

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, "folio_stats_snapshot_") && strstr(entry->d_name, ".log")) {
            char filepath[1024];
            if (snprintf(filepath, sizeof(filepath), "%s/%s", LOG_DIR, entry->d_name) >= sizeof(filepath)) {
                fprintf(stderr, "Error: File path too long: %s/%s\n", LOG_DIR, entry->d_name);
                continue;
            }

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

    printf("Integrated data saved to %s\n", OUTPUT_FILE);
}

int main() {
    integrate_snapshots();
    return EXIT_SUCCESS;
}