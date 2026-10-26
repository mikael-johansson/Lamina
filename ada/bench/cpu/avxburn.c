// avxburn <seconds>: maddubs/madd/add chains like the expert kernels' inner loop, on the calling thread (pin it)
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
int main(int argc, char** argv) {
    const double secs = argc > 1 ? atof(argv[1]) : 10;
    __m256i a[8], w = _mm256_set1_epi8(3), x = _mm256_set1_epi8(5), s = _mm256_set1_epi16(1);
    for (int i = 0; i < 8; ++i) a[i] = _mm256_set1_epi32(i);
    struct timespec t0, t;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long n = 0;
    do {
        for (int k = 0; k < 4096; ++k)
            for (int i = 0; i < 8; ++i) a[i] = _mm256_add_epi32(a[i], _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), s));
        n += 4096;
        clock_gettime(CLOCK_MONOTONIC, &t);
    } while ((t.tv_sec - t0.tv_sec) + (t.tv_nsec - t0.tv_nsec) * 1e-9 < secs);
    int r = 0;
    for (int i = 0; i < 8; ++i) r += _mm256_extract_epi32(a[i], 0);
    printf("%d\n", r & 1);
    return 0;
}
