#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>

#define LOG_DIR "folio_logs"          // 고정된 로그 디렉토리 경로
#define OUTPUT_FILE "integrated_data.csv" // 고정된 출력 CSV 파일 이름

int compare_filenames(const void *a, const void *b) {
    // 파일 이름에서 스냅샷 번호를 추출하여 비교
    const char *file_a = *(const char **)a;
    const char *file_b = *(const char **)b;

    int snapshot_a = 0, snapshot_b = 0;
    sscanf(file_a, "folio_stats_snapshot_%d.log", &snapshot_a);
    sscanf(file_b, "folio_stats_snapshot_%d.log", &snapshot_b);

    return snapshot_a - snapshot_b;
}

void integrate_snapshots() {
    DIR *dir = opendir(LOG_DIR);
    if (!dir) {
        perror("Error opening log directory");
        exit(EXIT_FAILURE);
    }

    // 파일 이름을 저장할 배열
    char *filenames[1024];
    int file_count = 0;

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strstr(entry->d_name, "folio_stats_snapshot_") && strstr(entry->d_name, ".log")) {
            filenames[file_count] = strdup(entry->d_name);
            if (!filenames[file_count]) {
                perror("Error allocating memory for filename");
                closedir(dir);
                exit(EXIT_FAILURE);
            }
            file_count++;
        }
    }
    closedir(dir);

    // 파일 이름 정렬
    qsort(filenames, file_count, sizeof(char *), compare_filenames);

    FILE *output = fopen(OUTPUT_FILE, "w");
    if (!output) {
        perror("Error opening output file");
        for (int i = 0; i < file_count; i++) {
            free(filenames[i]);
        }
        exit(EXIT_FAILURE);
    }

    fprintf(output, "snapshot,node,pfn,source_nid,migrate_count\n"); // CSV 헤더

    // 정렬된 파일 이름을 순서대로 처리
    for (int i = 0; i < file_count; i++) {
        char filepath[1024];
        snprintf(filepath, sizeof(filepath), "%s/%s", LOG_DIR, filenames[i]);

        FILE *file = fopen(filepath, "r");
        if (!file) {
            perror("Error opening snapshot file");
            free(filenames[i]);
            continue;
        }

        char line[256];
        int snapshot = 0;

        // 스냅샷 번호 추출
        sscanf(filenames[i], "folio_stats_snapshot_%d.log", &snapshot);

        // 파일 내용 읽기
        while (fgets(line, sizeof(line), file)) {
            int node, pfn, source_nid, migrate_count;
            if (sscanf(line, "%d,%d,%d,%d", &node, &pfn, &source_nid, &migrate_count) == 4) {
                fprintf(output, "%d,%d,%d,%d,%d\n", snapshot, node, pfn, source_nid, migrate_count);
            }
        }

        fclose(file);
        free(filenames[i]);
    }

    fclose(output);

    printf("Integrated data saved to %s\n", OUTPUT_FILE);
}

int main() {
    integrate_snapshots();
    return EXIT_SUCCESS;
}