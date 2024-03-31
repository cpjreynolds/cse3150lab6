#ifdef TESTING
#include "doctest.h"
#include <vector>

#include <sstream>

#include "mtestvals.hpp"
#include "matrix.hpp"

// tests implementations that are common to all matrix types
//
// essentially everything but multiplication and ensuring the correct padding on
// construction.
TEST_SUITE("matrix")
{
    TEST_CASE("operator>>")
    {
        std::istringstream input("1 2 3\n4 5 6\n7 8 9");
        basic_matrix tmat{{1., 2., 3.}, {4., 5., 6.}, {7., 8., 9.}};

        basic_matrix m;
        input >> m;

        CHECK(m == tmat);
    }

    TEST_CASE("equality")
    {
        // need to initialize two identical matrices in two different ways to
        // ensure we're not comparing bad data to bad data.
        //
        // furthermore, all the other tests rely on equality testing and
        // implicitly test it.
        auto a = naive_matrix({{1., 2., 3., 4.},
                               {5., 6., 7., 8.},
                               {9., 10., 11., 12.},
                               {13., 14., 15., 16.}});
        auto b = naive_matrix(4);
        for (auto i = 0u; i < 4; ++i) {
            for (auto j = 0u; j < 4; ++j) {
                b[i, j] = 4 * i + j + 1;
            }
        }
        CHECK(a == b);
    }

    TEST_CASE("uniform construction")
    {
        auto a = naive_matrix(4);

        for (auto i = 0u; i < 4; ++i) {
            for (auto j = 0u; j < 4; ++j) {
                CHECK(a[i, j] == 0.0);
            }
        }

        auto b = naive_matrix(4, 1.0);

        for (auto i = 0u; i < 4; ++i) {
            for (auto j = 0u; j < 4; ++j) {
                CHECK(b[i, j] == 1.0);
            }
        }
    }

    TEST_CASE("list and iterator construction")
    {
        auto a = naive_matrix({{1., 2., 3., 4.},
                               {5., 6., 7., 8.},
                               {9., 10., 11., 12.},
                               {13., 14., 15., 16.}});

        std::vector<float> bv(4 * 4);
        std::iota(bv.begin(), bv.end(), 1.0);

        auto b = naive_matrix(4, bv.begin(), bv.end());

        for (auto i = 0u; i < 4; ++i) {
            for (auto j = 0u; j < 4; ++j) {
                CHECK(a[i, j] == 4 * i + j + 1);
            }
        }
        for (auto i = 0u; i < 4; ++i) {
            for (auto j = 0u; j < 4; ++j) {
                CHECK(a[i, j] == b[i, j]);
            }
        }
        CHECK(a == b);
    }

    TEST_CASE("random")
    {
        // best I can do here is ensure all are within the bounds
        auto a = naive_matrix::random(4, 1.0, 2.0);
        for (auto i = 0u; i < 4; ++i) {
            for (auto j = 0u; j < 4; ++j) {
                CHECK(a[i, j] >= 1.0);
                CHECK(a[i, j] <= 2.0);
            }
        }
    }
}
TEST_SUITE("32-bit SISD")
{
    TEST_CASE("naive_matrix")
    {
        SUBCASE("m7x7_1")
        {
            const auto a = naive_matrix(mtest::m7x7_1_AB);
            const auto b = naive_matrix(mtest::m7x7_1_AB);
            const auto expect = naive_matrix(mtest::m7x7_1_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m7x7_2")
        {
            const auto a = naive_matrix(mtest::m7x7_2_AB);
            const auto b = naive_matrix(mtest::m7x7_2_AB);
            const auto expect = naive_matrix(mtest::m7x7_2_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m7x7")
        {
            const auto a = naive_matrix(mtest::m7x7_A);
            const auto b = naive_matrix(mtest::m7x7_B);
            const auto expect = naive_matrix(mtest::m7x7_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m13x13_1")
        {
            const auto a = naive_matrix(mtest::m13x13_1_AB);
            const auto b = naive_matrix(mtest::m13x13_1_AB);
            const auto expect = naive_matrix(mtest::m13x13_1_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m15x15")
        {
            const auto a = naive_matrix(mtest::m15x15_A);
            const auto b = naive_matrix(mtest::m15x15_B);
            const auto expect = naive_matrix(mtest::m15x15_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m16x16")
        {
            const auto a = naive_matrix(mtest::m16x16_A);
            const auto b = naive_matrix(mtest::m16x16_B);
            const auto expect = naive_matrix(mtest::m16x16_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m32x32")
        {
            const auto a = naive_matrix(mtest::m32x32_A);
            const auto b = naive_matrix(mtest::m32x32_B);
            const auto expect = naive_matrix(mtest::m32x32_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m64x64")
        {
            const auto a = naive_matrix(mtest::m64x64_A);
            const auto b = naive_matrix(mtest::m64x64_B);
            const auto expect = naive_matrix(mtest::m64x64_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("stranke94")
        {
            const auto a = naive_matrix(mtest::stranke94);
            const auto b = naive_matrix(mtest::stranke94);
            const auto expect = naive_matrix(mtest::stranke94_squared);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("bcsstk02")
        {
            const auto a = naive_matrix(mtest::bcsstk02);
            const auto b = naive_matrix(mtest::bcsstk02);
            const auto expect = naive_matrix(mtest::bcsstk02_squared);

            const auto c = a * b;

            // because of the insane range of floats in the
            // matrix, FPE propagation really fucks me on this one,
            // so I give a little more leeway
            for (auto i = 0u; i < a.dim(); ++i) {
                for (auto j = 0u; j < a.dim(); ++j) {
                    CHECK(c[i, j] ==
                          doctest::Approx(expect[i, j]).epsilon(0.005));
                }
            }
        }
    }
}

// f32x4
TEST_SUITE("128-bit SIMD")
{
    TEST_CASE("simd_matrix")
    {
        SUBCASE("m7x7_1")
        {
            const auto a = simd_matrix<m128>(mtest::m7x7_1_AB);
            const auto b = simd_matrix<m128>(mtest::m7x7_1_AB);
            const auto expect = simd_matrix<m128>(mtest::m7x7_1_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m7x7_2")
        {
            const auto a = simd_matrix<m128>(mtest::m7x7_2_AB);
            const auto b = simd_matrix<m128>(mtest::m7x7_2_AB);
            const auto expect = simd_matrix<m128>(mtest::m7x7_2_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m7x7")
        {
            const auto a = simd_matrix<m128>(mtest::m7x7_A);
            const auto b = simd_matrix<m128>(mtest::m7x7_B);
            const auto expect = simd_matrix<m128>(mtest::m7x7_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m13x13_1")
        {
            const auto a = simd_matrix<m128>(mtest::m13x13_1_AB);
            const auto b = simd_matrix<m128>(mtest::m13x13_1_AB);
            const auto expect = simd_matrix<m128>(mtest::m13x13_1_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m15x15")
        {
            const auto a = simd_matrix<m128>(mtest::m15x15_A);
            const auto b = simd_matrix<m128>(mtest::m15x15_B);
            const auto expect = simd_matrix<m128>(mtest::m15x15_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m16x16")
        {
            const auto a = simd_matrix<m128>(mtest::m16x16_A);
            const auto b = simd_matrix<m128>(mtest::m16x16_B);
            const auto expect = simd_matrix<m128>(mtest::m16x16_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m32x32")
        {
            const auto a = simd_matrix<m128>(mtest::m32x32_A);
            const auto b = simd_matrix<m128>(mtest::m32x32_B);
            const auto expect = simd_matrix<m128>(mtest::m32x32_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m64x64")
        {
            const auto a = simd_matrix<m128>(mtest::m64x64_A);
            const auto b = simd_matrix<m128>(mtest::m64x64_B);
            const auto expect = simd_matrix<m128>(mtest::m64x64_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("stranke94")
        {
            const auto a = simd_matrix<m128>(mtest::stranke94);
            const auto b = simd_matrix<m128>(mtest::stranke94);
            const auto expect = simd_matrix<m128>(mtest::stranke94_squared);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("bcsstk02")
        {
            const auto a = simd_matrix<m128>(mtest::bcsstk02);
            const auto b = simd_matrix<m128>(mtest::bcsstk02);
            const auto expect = simd_matrix<m128>(mtest::bcsstk02_squared);

            const auto c = a * b;

            for (auto i = 0u; i < a.dim(); ++i) {
                for (auto j = 0u; j < a.dim(); ++j) {
                    CHECK(c[i, j] ==
                          doctest::Approx(expect[i, j]).epsilon(0.005));
                }
            }
        }
    }

    TEST_CASE("kernel_matrix")
    {
        SUBCASE("m7x7_1")
        {
            const auto a = kernel_matrix<m128>(mtest::m7x7_1_AB);
            const auto b = kernel_matrix<m128>(mtest::m7x7_1_AB);
            const auto expect = kernel_matrix<m128>(mtest::m7x7_1_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m7x7_2")
        {
            const auto a = kernel_matrix<m128>(mtest::m7x7_2_AB);
            const auto b = kernel_matrix<m128>(mtest::m7x7_2_AB);
            const auto expect = kernel_matrix<m128>(mtest::m7x7_2_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m7x7")
        {
            const auto a = kernel_matrix<m128>(mtest::m7x7_A);
            const auto b = kernel_matrix<m128>(mtest::m7x7_B);
            const auto expect = kernel_matrix<m128>(mtest::m7x7_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m13x13_1")
        {
            const auto a = kernel_matrix<m128>(mtest::m13x13_1_AB);
            const auto b = kernel_matrix<m128>(mtest::m13x13_1_AB);
            const auto expect = kernel_matrix<m128>(mtest::m13x13_1_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m15x15")
        {
            const auto a = kernel_matrix<m128>(mtest::m15x15_A);
            const auto b = kernel_matrix<m128>(mtest::m15x15_B);
            const auto expect = kernel_matrix<m128>(mtest::m15x15_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m16x16")
        {
            const auto a = kernel_matrix<m128>(mtest::m16x16_A);
            const auto b = kernel_matrix<m128>(mtest::m16x16_B);
            const auto expect = kernel_matrix<m128>(mtest::m16x16_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m32x32")
        {
            const auto a = kernel_matrix<m128>(mtest::m32x32_A);
            const auto b = kernel_matrix<m128>(mtest::m32x32_B);
            const auto expect = kernel_matrix<m128>(mtest::m32x32_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m64x64")
        {
            const auto a = kernel_matrix<m128>(mtest::m64x64_A);
            const auto b = kernel_matrix<m128>(mtest::m64x64_B);
            const auto expect = kernel_matrix<m128>(mtest::m64x64_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("stranke94")
        {
            const auto a = kernel_matrix<m128>(mtest::stranke94);
            const auto b = kernel_matrix<m128>(mtest::stranke94);
            const auto expect = kernel_matrix<m128>(mtest::stranke94_squared);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("bcsstk02")
        {
            const auto a = kernel_matrix<m128>(mtest::bcsstk02);
            const auto b = kernel_matrix<m128>(mtest::bcsstk02);
            const auto expect = kernel_matrix<m128>(mtest::bcsstk02_squared);

            const auto c = a * b;

            for (auto i = 0u; i < a.dim(); ++i) {
                for (auto j = 0u; j < a.dim(); ++j) {
                    CHECK(c[i, j] ==
                          doctest::Approx(expect[i, j]).epsilon(0.005));
                }
            }
        }
    }
}

// f32x8
#ifdef __AVX2__
TEST_SUITE("256-bit SIMD")
{
    TEST_CASE("simd_matrix")
    {
        SUBCASE("m7x7_1")
        {
            const auto a = simd_matrix<m256>(mtest::m7x7_1_AB);
            const auto b = simd_matrix<m256>(mtest::m7x7_1_AB);
            const auto expect = simd_matrix<m256>(mtest::m7x7_1_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m7x7_2")
        {
            const auto a = simd_matrix<m256>(mtest::m7x7_2_AB);
            const auto b = simd_matrix<m256>(mtest::m7x7_2_AB);
            const auto expect = simd_matrix<m256>(mtest::m7x7_2_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m7x7")
        {
            const auto a = simd_matrix<m256>(mtest::m7x7_A);
            const auto b = simd_matrix<m256>(mtest::m7x7_B);
            const auto expect = simd_matrix<m256>(mtest::m7x7_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m13x13_1")
        {
            const auto a = simd_matrix<m256>(mtest::m13x13_1_AB);
            const auto b = simd_matrix<m256>(mtest::m13x13_1_AB);
            const auto expect = simd_matrix<m256>(mtest::m13x13_1_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m15x15")
        {
            const auto a = simd_matrix<m256>(mtest::m15x15_A);
            const auto b = simd_matrix<m256>(mtest::m15x15_B);
            const auto expect = simd_matrix<m256>(mtest::m15x15_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m16x16")
        {
            const auto a = simd_matrix<m256>(mtest::m16x16_A);
            const auto b = simd_matrix<m256>(mtest::m16x16_B);
            const auto expect = simd_matrix<m256>(mtest::m16x16_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m32x32")
        {
            const auto a = simd_matrix<m256>(mtest::m32x32_A);
            const auto b = simd_matrix<m256>(mtest::m32x32_B);
            const auto expect = simd_matrix<m256>(mtest::m32x32_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m64x64")
        {
            const auto a = simd_matrix<m256>(mtest::m64x64_A);
            const auto b = simd_matrix<m256>(mtest::m64x64_B);
            const auto expect = simd_matrix<m256>(mtest::m64x64_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("stranke94")
        {
            const auto a = simd_matrix<m256>(mtest::stranke94);
            const auto b = simd_matrix<m256>(mtest::stranke94);
            const auto expect = simd_matrix<m256>(mtest::stranke94_squared);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("bcsstk02")
        {
            const auto a = simd_matrix<m256>(mtest::bcsstk02);
            const auto b = simd_matrix<m256>(mtest::bcsstk02);
            const auto expect = simd_matrix<m256>(mtest::bcsstk02_squared);

            const auto c = a * b;

            for (auto i = 0u; i < a.dim(); ++i) {
                for (auto j = 0u; j < a.dim(); ++j) {
                    CHECK(c[i, j] ==
                          doctest::Approx(expect[i, j]).epsilon(0.005));
                }
            }
        }
    }

    TEST_CASE("kernel_matrix")
    {
        SUBCASE("m7x7_1")
        {
            const auto a = kernel_matrix<m256>(mtest::m7x7_1_AB);
            const auto b = kernel_matrix<m256>(mtest::m7x7_1_AB);
            const auto expect = kernel_matrix<m256>(mtest::m7x7_1_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m7x7_2")
        {
            const auto a = kernel_matrix<m256>(mtest::m7x7_2_AB);
            const auto b = kernel_matrix<m256>(mtest::m7x7_2_AB);
            const auto expect = kernel_matrix<m256>(mtest::m7x7_2_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m7x7")
        {
            const auto a = kernel_matrix<m256>(mtest::m7x7_A);
            const auto b = kernel_matrix<m256>(mtest::m7x7_B);
            const auto expect = kernel_matrix<m256>(mtest::m7x7_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m13x13_1")
        {
            const auto a = kernel_matrix<m256>(mtest::m13x13_1_AB);
            const auto b = kernel_matrix<m256>(mtest::m13x13_1_AB);
            const auto expect = kernel_matrix<m256>(mtest::m13x13_1_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m15x15")
        {
            const auto a = kernel_matrix<m256>(mtest::m15x15_A);
            const auto b = kernel_matrix<m256>(mtest::m15x15_B);
            const auto expect = kernel_matrix<m256>(mtest::m15x15_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m16x16")
        {
            const auto a = kernel_matrix<m256>(mtest::m16x16_A);
            const auto b = kernel_matrix<m256>(mtest::m16x16_B);
            const auto expect = kernel_matrix<m256>(mtest::m16x16_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m32x32")
        {
            const auto a = kernel_matrix<m256>(mtest::m32x32_A);
            const auto b = kernel_matrix<m256>(mtest::m32x32_B);
            const auto expect = kernel_matrix<m256>(mtest::m32x32_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m64x64")
        {
            const auto a = kernel_matrix<m256>(mtest::m64x64_A);
            const auto b = kernel_matrix<m256>(mtest::m64x64_B);
            const auto expect = kernel_matrix<m256>(mtest::m64x64_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("stranke94")
        {
            const auto a = kernel_matrix<m256>(mtest::stranke94);
            const auto b = kernel_matrix<m256>(mtest::stranke94);
            const auto expect = kernel_matrix<m256>(mtest::stranke94_squared);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("bcsstk02")
        {
            const auto a = kernel_matrix<m256>(mtest::bcsstk02);
            const auto b = kernel_matrix<m256>(mtest::bcsstk02);
            const auto expect = kernel_matrix<m256>(mtest::bcsstk02_squared);

            const auto c = a * b;

            for (auto i = 0u; i < a.dim(); ++i) {
                for (auto j = 0u; j < a.dim(); ++j) {
                    CHECK(c[i, j] ==
                          doctest::Approx(expect[i, j]).epsilon(0.005));
                }
            }
        }
    }
}
#endif

// f32x16
#ifdef __AVX512F__
TEST_SUITE("512-bit SIMD")
{
    TEST_CASE("simd_matrix")
    {
        SUBCASE("m7x7_1")
        {
            const auto a = simd_matrix<m512>(mtest::m7x7_1_AB);
            const auto b = simd_matrix<m512>(mtest::m7x7_1_AB);
            const auto expect = simd_matrix<m512>(mtest::m7x7_1_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m7x7_2")
        {
            const auto a = simd_matrix<m512>(mtest::m7x7_2_AB);
            const auto b = simd_matrix<m512>(mtest::m7x7_2_AB);
            const auto expect = simd_matrix<m512>(mtest::m7x7_2_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m7x7")
        {
            const auto a = simd_matrix<m512>(mtest::m7x7_A);
            const auto b = simd_matrix<m512>(mtest::m7x7_B);
            const auto expect = simd_matrix<m512>(mtest::m7x7_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m13x13_1")
        {
            const auto a = simd_matrix<m512>(mtest::m13x13_1_AB);
            const auto b = simd_matrix<m512>(mtest::m13x13_1_AB);
            const auto expect = simd_matrix<m512>(mtest::m13x13_1_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m15x15")
        {
            const auto a = simd_matrix<m512>(mtest::m15x15_A);
            const auto b = simd_matrix<m512>(mtest::m15x15_B);
            const auto expect = simd_matrix<m512>(mtest::m15x15_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m16x16")
        {
            const auto a = simd_matrix<m512>(mtest::m16x16_A);
            const auto b = simd_matrix<m512>(mtest::m16x16_B);
            const auto expect = simd_matrix<m512>(mtest::m16x16_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m32x32")
        {
            const auto a = simd_matrix<m512>(mtest::m32x32_A);
            const auto b = simd_matrix<m512>(mtest::m32x32_B);
            const auto expect = simd_matrix<m512>(mtest::m32x32_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m64x64")
        {
            const auto a = simd_matrix<m512>(mtest::m64x64_A);
            const auto b = simd_matrix<m512>(mtest::m64x64_B);
            const auto expect = simd_matrix<m512>(mtest::m64x64_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("stranke94")
        {
            const auto a = simd_matrix<m512>(mtest::stranke94);
            const auto b = simd_matrix<m512>(mtest::stranke94);
            const auto expect = simd_matrix<m512>(mtest::stranke94_squared);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("bcsstk02")
        {
            const auto a = simd_matrix<m512>(mtest::bcsstk02);
            const auto b = simd_matrix<m512>(mtest::bcsstk02);
            const auto expect = simd_matrix<m512>(mtest::bcsstk02_squared);

            const auto c = a * b;

            for (auto i = 0u; i < a.dim(); ++i) {
                for (auto j = 0u; j < a.dim(); ++j) {
                    CHECK(c[i, j] ==
                          doctest::Approx(expect[i, j]).epsilon(0.005));
                }
            }
        }
    }

    TEST_CASE("kernel_matrix")
    {
        SUBCASE("m7x7_1")
        {
            const auto a = kernel_matrix<m512>(mtest::m7x7_1_AB);
            const auto b = kernel_matrix<m512>(mtest::m7x7_1_AB);
            const auto expect = kernel_matrix<m512>(mtest::m7x7_1_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m7x7_2")
        {
            const auto a = kernel_matrix<m512>(mtest::m7x7_2_AB);
            const auto b = kernel_matrix<m512>(mtest::m7x7_2_AB);
            const auto expect = kernel_matrix<m512>(mtest::m7x7_2_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m7x7")
        {
            const auto a = kernel_matrix<m512>(mtest::m7x7_A);
            const auto b = kernel_matrix<m512>(mtest::m7x7_B);
            const auto expect = kernel_matrix<m512>(mtest::m7x7_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m13x13_1")
        {
            const auto a = kernel_matrix<m512>(mtest::m13x13_1_AB);
            const auto b = kernel_matrix<m512>(mtest::m13x13_1_AB);
            const auto expect = kernel_matrix<m512>(mtest::m13x13_1_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m15x15")
        {
            const auto a = kernel_matrix<m512>(mtest::m15x15_A);
            const auto b = kernel_matrix<m512>(mtest::m15x15_B);
            const auto expect = kernel_matrix<m512>(mtest::m15x15_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m16x16")
        {
            const auto a = kernel_matrix<m512>(mtest::m16x16_A);
            const auto b = kernel_matrix<m512>(mtest::m16x16_B);
            const auto expect = kernel_matrix<m512>(mtest::m16x16_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m32x32")
        {
            const auto a = kernel_matrix<m512>(mtest::m32x32_A);
            const auto b = kernel_matrix<m512>(mtest::m32x32_B);
            const auto expect = kernel_matrix<m512>(mtest::m32x32_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("m64x64")
        {
            const auto a = kernel_matrix<m512>(mtest::m64x64_A);
            const auto b = kernel_matrix<m512>(mtest::m64x64_B);
            const auto expect = kernel_matrix<m512>(mtest::m64x64_C);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("stranke94")
        {
            const auto a = kernel_matrix<m512>(mtest::stranke94);
            const auto b = kernel_matrix<m512>(mtest::stranke94);
            const auto expect = kernel_matrix<m512>(mtest::stranke94_squared);

            const auto c = a * b;

            CHECK(c == expect);
        }
        SUBCASE("bcsstk02")
        {
            const auto a = kernel_matrix<m512>(mtest::bcsstk02);
            const auto b = kernel_matrix<m512>(mtest::bcsstk02);
            const auto expect = kernel_matrix<m512>(mtest::bcsstk02_squared);

            const auto c = a * b;

            for (auto i = 0u; i < a.dim(); ++i) {
                for (auto j = 0u; j < a.dim(); ++j) {
                    CHECK(c[i, j] ==
                          doctest::Approx(expect[i, j]).epsilon(0.005));
                }
            }
        }
    }
}
#endif

#endif
