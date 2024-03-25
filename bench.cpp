#include "matrix.hpp"
#include "instrument.hpp"

#define AVXTYPES simd_matrix<m128>, kernel_matrix<m128>

#define AVX2TYPES AVXTYPES, simd_matrix<m256>, kernel_matrix<m256>

#define AVX512TYPES                                                            \
    AVX2TYPES, simd_matrix<m512>, kernel_matrix<m512>, kernel_matrix<m512, 5, 5>

#if defined(__AVX512F__)
#define MAT_TYPES AVX512TYPES
#elif defined(__AVX2__)
#define MAT_TYPES AVX2TYPES
#else
#define MAT_TYPES AVXTYPES
#endif

#define MTYPES                                                                 \
    kernel_matrix<m512, 6, 2>, kernel_matrix<m512, 8, 2>,                      \
        kernel_matrix<m512, 12, 2>, kernel_matrix<m512, 6, 4>

int main()
{
    // auto [l1, l2, l3] = cachesize();

    benchmarks benches;

    benches.add<MTYPES>(2048);
    benches.add<MTYPES>(1024);
    benches.add<MTYPES>(512);
    benches.add<MTYPES>(256);
    benches.add<MTYPES>(128);

    benches.run();
}
