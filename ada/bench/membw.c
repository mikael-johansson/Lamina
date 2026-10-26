// Read bandwidth per physical address range (needs root for /proc/self/pagemap).
// Build: gcc -O2 -mavx2 -pthread -o membw membw.c
// Usage: sudo ./membw [GiB to allocate] [threads]
// Pages are grouped into 1 GiB bins by physical address; each bin with enough pages is read by all threads
// and its GB/s printed, so interleaved (dual-channel) and single-channel ranges show up as two levels.
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>

#define PAGE 4096UL
#define BIN_SHIFT 30

typedef struct { char **pages; size_t n; double sum; } job_t;

static void *reader(void *p) {
    job_t *j = p;
    __m256i acc = _mm256_setzero_si256();
    for (size_t i = 0; i < j->n; i++) {
        const __m256i *q = (const __m256i *)j->pages[i];
        for (int k = 0; k < (int)(PAGE / 32); k += 4) {
            acc = _mm256_add_epi64(acc, _mm256_load_si256(q + k));
            acc = _mm256_add_epi64(acc, _mm256_load_si256(q + k + 1));
            acc = _mm256_add_epi64(acc, _mm256_load_si256(q + k + 2));
            acc = _mm256_add_epi64(acc, _mm256_load_si256(q + k + 3));
        }
    }
    long long out[4];
    _mm256_storeu_si256((__m256i *)out, acc);
    j->sum = (double)(out[0] + out[1] + out[2] + out[3]);
    return NULL;
}

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

// Read `n` pages with `nt` threads, `reps` times; return GB/s.
static double measure(char **pages, size_t n, int nt, int reps) {
    pthread_t th[64];
    job_t jobs[64];
    double t0 = now();
    for (int r = 0; r < reps; r++) {
        for (int t = 0; t < nt; t++) {
            size_t a = n * t / nt, b = n * (t + 1) / nt;
            jobs[t] = (job_t){pages + a, b - a, 0};
            pthread_create(&th[t], NULL, reader, &jobs[t]);
        }
        for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
    }
    return (double)n * PAGE * reps / (now() - t0) / 1e9;
}

int main(int argc, char **argv) {
    size_t gib = argc > 1 ? strtoul(argv[1], NULL, 10) : 8;
    int nt = argc > 2 ? atoi(argv[2]) : 8;
    size_t bytes = gib << 30, npages = bytes / PAGE;

    char *buf = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) { perror("mmap"); return 1; }
    madvise(buf, bytes, MADV_HUGEPAGE);
    for (size_t i = 0; i < bytes; i += PAGE) buf[i] = (char)i;   // fault every page in
    memset(buf, 1, bytes);

    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) { perror("pagemap"); return 1; }
    uint64_t *map = malloc(npages * 8);
    if (pread(fd, map, npages * 8, (off_t)((uintptr_t)buf / PAGE) * 8) != (ssize_t)(npages * 8)) {
        perror("pread pagemap"); return 1;
    }
    close(fd);

    size_t nbins = 0;
    for (size_t i = 0; i < npages; i++) {
        uint64_t pfn = map[i] & ((1ULL << 55) - 1);
        if (!(map[i] >> 63) || pfn == 0) { fprintf(stderr, "no physical addresses: run as root\n"); return 1; }
        size_t b = (pfn * PAGE) >> BIN_SHIFT;
        if (b + 1 > nbins) nbins = b + 1;
    }
    size_t *cnt = calloc(nbins, sizeof *cnt);
    for (size_t i = 0; i < npages; i++) cnt[((map[i] & ((1ULL << 55) - 1)) * PAGE) >> BIN_SHIFT]++;
    char ***lists = calloc(nbins, sizeof *lists);
    size_t *fill = calloc(nbins, sizeof *fill);
    for (size_t b = 0; b < nbins; b++) if (cnt[b]) lists[b] = malloc(cnt[b] * sizeof(char *));
    for (size_t i = 0; i < npages; i++) {
        size_t b = ((map[i] & ((1ULL << 55) - 1)) * PAGE) >> BIN_SHIFT;
        lists[b][fill[b]++] = buf + i * PAGE;
    }

    char **all = malloc(npages * sizeof(char *));
    for (size_t i = 0; i < npages; i++) all[i] = buf + i * PAGE;
    printf("all %zu GiB, %d threads: %.1f GB/s\n", gib, nt, measure(all, npages, nt, 3));

    size_t min_pages = (256UL << 20) / PAGE;   // bins with at least 256 MiB of our pages
    printf("phys GiB   MiB here   GB/s\n");
    for (size_t b = 0; b < nbins; b++) {
        if (cnt[b] < min_pages) continue;
        int reps = (int)((8UL << 30) / (cnt[b] * PAGE)) + 1;   // read ~8 GB per bin
        printf("%4zu-%-4zu %9zu %7.1f\n", b, b + 1, cnt[b] * PAGE >> 20, measure(lists[b], cnt[b], nt, reps));
    }
    return 0;
}
