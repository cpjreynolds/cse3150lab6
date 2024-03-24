#include "matrix.hpp"
#include "instrument.hpp"

#define AVXTYPES simd_matrix<m128>, kernel_matrix<m128>

#define AVX2TYPES AVXTYPES, simd_matrix<m256>, kernel_matrix<m256>

#define AVX512TYPES AVX2TYPES, simd_matrix<m512>, kernel_matrix<m512>

#if defined(__AVX512F__)
#define MAT_TYPES AVX512TYPES
#elif defined(__AVX2__)
#define MAT_TYPES AVX2TYPES
#else
#define MAT_TYPES AVXTYPES
#endif

int main()
{
    // auto [l1, l2, l3] = cachesize();

    benchmarks benches;

    benches.add<MAT_TYPES>(2048);
    benches.add<MAT_TYPES>(1024);
    benches.add<MAT_TYPES, naive_matrix>(512);
    benches.add<MAT_TYPES, naive_matrix>(256);
    benches.add<MAT_TYPES, naive_matrix>(128);

    benches.run();
}
