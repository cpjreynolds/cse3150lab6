/*
 * Graders beware. Here be dragons.
 *
 * (I had a lot of fun)
 */
#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <initializer_list>
#include <random>
#include <utility>
#include <algorithm>
#include <version>
#include <sstream>

#include "cpuinfo.hpp"

// summon the x86-64 gremlins.
#include <immintrin.h>

#ifndef __AVX__
#error "matrix.hpp requires at least AVX1 support"
#endif

// the unions stop the compiler from discarding attributes declared on the
// intrinsic type when used as a template argument.
//
// Namely __attribute__((alignment(x), vector_size(y)))
union m128 {
    __m128 xmm; // actual hardware register
    using type = __m128;
};
static_assert(sizeof(m128) == alignof(m128));

#ifdef __AVX2__
union m256 {
    __m256 ymm;
    using type = __m256;
};
static_assert(sizeof(m256) == alignof(m256));
#endif

#ifdef __AVX512F__
union m512 {
    __m512 zmm;
    using type = __m512;
};
static_assert(sizeof(m512) == alignof(m512));
#endif

// types that can be used as vectors (vec-able)
template<typename T>
concept vecable = std::same_as<T, m128>
#ifdef __AVX2__
                  || std::same_as<T, m256>
#endif
#ifdef __AVX512F__
                  || std::same_as<T, m512>
#endif
    ;

// floating-point comparison
//
// https://floating-point-gui.de/errors/comparison/
[[maybe_unused]] constexpr bool
cmpf(float a, float b, float eps = std::numeric_limits<float>::epsilon() * 100)
{
    constexpr float min_norm = std::numeric_limits<float>::min();
    constexpr float max_val = std::numeric_limits<float>::max();

    float abs_a = std::fabs(a);
    float abs_b = std::fabs(b);
    float diff = std::fabs(a - b);

    if (a == b) { // shortcut, handles infinities
        return true;
    }
    else if (a == 0 || b == 0 || (abs_a + abs_b < min_norm)) {
        // a or b is zero or both are extremely close to it
        // relative error is less meaningful here
        return diff < (eps * min_norm);
    }
    else { // use relative error
        return diff / std::min((abs_a + abs_b), max_val) < eps;
    }
}

template<typename T>
struct vec_t_helper {
    using type = T::type;
};

template<>
struct vec_t_helper<float> {
    using type = float;
};

// Uses CRTP for static polymorphism to determine multiplication behaviour.
//
// https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
//
// https://en.cppreference.com/w/cpp/language/crtp
template<typename T, typename V = m128>
    requires vecable<V> || std::same_as<V, float>
class matrix {
public:
    using vec_t = vec_t_helper<V>::type;

    matrix() : dim_{0}, data_{nullptr, &matrix::free} {}

    explicit matrix(size_t n)
        : dim_{n}, data_{matrix::allocate((*this)->bytes()), &matrix::free}
    {
        // I actually needed this during testing. large alignment
        // values cause aligned_alloc to return null.
        //
        // using _mm_malloc() instead for now.
        if (data() == nullptr) {
            throw std::bad_alloc();
        }
        std::memset(data(), 0, (*this)->bytes());
    };

    matrix(size_t n, float init) : matrix{n}
    {
        for (auto i = 0u; i < n; ++i) {
            std::fill_n((*this)[i], n, init);
        }
    }

    // downcast for compile-time polymorphism.
    //
    // (*this)->xyz() NOT this->xyz() for overrides
    T* operator->() { return static_cast<T*>(this); }
    const T* operator->() const { return static_cast<const T*>(this); }

    matrix(std::initializer_list<std::initializer_list<float>> init)
        : matrix(init.size())
    {
        if (init.size() != init.begin()->size()) {
            throw std::logic_error("mismatched matrix dimensions");
        }
        size_t i = 0;
        for (auto row : init) {
            std::memcpy((*this)[i], row.begin(), dim() * sizeof(float));
            ++i;
        }
    }

    matrix(std::vector<std::vector<float>>& init) : matrix{init.size()}
    {
        if (init.size() != init.begin()->size()) {
            throw std::logic_error("mismatched matrix dimensions");
        }
        size_t i = 0;
        for (auto row : init) {
            std::memcpy((*this)[i], row.data(), dim() * sizeof(float));
            ++i;
        }
    }

    template<std::input_iterator I>
        requires std::same_as<std::iter_value_t<I>, float>
    matrix(size_t n, I fst, I lst) : matrix(n)
    {
        if (std::distance(fst, lst) != ptrdiff_t(n * n)) {
            throw std::logic_error("mismatched matrix dimensions");
        }
        for (auto i = 0u; i < n; ++i) {
            for (auto j = 0u; j < n; ++j) {
                (*this)[i, j] = *fst++;
            }
        }
    }

    // return a matrix filled with random floats
    static matrix random(size_t n, float a = 0.0, float b = 1.0)
    {
        matrix mat(n);
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dist(a, b);
        for (size_t i = 0; i < n; ++i) {
            std::generate_n(mat[i], n, [&] { return dist(gen); });
        }
        return mat;
    }

    // no copying, but moving is chill.
    matrix(const matrix&) = delete;
    matrix& operator=(const matrix&) = delete;
    matrix(matrix&&) = default;
    matrix& operator=(matrix&&) = default;

    friend bool operator==(const matrix& lhs, const matrix& rhs)
    {
        auto n = lhs.dim();
        if (n != rhs.dim()) {
            throw std::logic_error("mismatched matrix dimensions");
        }
        for (auto i = 0u; i < n; ++i) {
            for (auto j = 0u; j < n; ++j) {
                if (!cmpf(lhs[i, j], rhs[i, j])) {
                    return false;
                }
            }
        }
        return true;
    }

    friend bool operator!=(const matrix& lhs, const matrix& rhs)
    {
        return !(lhs == rhs);
    }

    // get a row
    float* operator[](size_t r) { return data() + (row_sizef() * r); }
    const float* operator[](size_t r) const
    {
        return data() + (row_sizef() * r);
    }

    // get an entry
    //
    // woohoo c++23 multidimensional indexing
    float& operator[](size_t r, size_t c)
    {
        return data()[(row_sizef() * r) + c];
    }

    const float& operator[](size_t r, size_t c) const
    {
        return data()[(row_sizef() * r) + c];
    }

    // be very careful with indices.
    //
    // The column index is always the same between vec_t/float indexing.
    //
    // The row in vec_t indexing is divided by vsize()
    vec_t& operator[](std::pair<size_t, size_t> idx)
    {
        // very legal and very cool
        vec_t* d = reinterpret_cast<vec_t*>(data());
        auto const rsv = (*this)->row_sizev();
        return d[rsv * idx.first + idx.second];
    }

    const vec_t& operator[](std::pair<size_t, size_t> idx) const
    {
        // undefined behaviour? what undefined behaviour?
        const vec_t* d = reinterpret_cast<const vec_t*>(data());
        auto const rsv = (*this)->row_sizev();
        return d[rsv * idx.first + idx.second];
    }

    // static polymorphism!
    matrix operator*(const matrix& rhs) const
    {
        return static_cast<const T*>(this)->operator*(rhs);
    }

    float* data() { return data_.get(); }
    const float* data() const { return data_.get(); }

    vec_t* vdata() { return reinterpret_cast<vec_t*>(data()); }
    const vec_t* vdata() const
    {
        return reinterpret_cast<const vec_t*>(data());
    }

    friend std::ostream& operator<<(std::ostream& os, const matrix& self)
    {
        os << '\n';
        for (size_t i = 0; i < self.dim(); ++i) {
            for (size_t j = 0; j < self.dim(); ++j) {
                os << std::setw(5) << std::fixed << std::setprecision(1)
                   << self[i, j] << ' ';
            }
            os << '\n';
        }
        os << std::setprecision(6);
        return os;
    }

    friend std::istream& operator>>(std::istream& is, matrix& self)
    {
        std::vector<std::vector<float>> vals;

        for (std::string line; std::getline(is, line);) {
            std::istringstream iss{std::move(line)};
            std::vector<float> linevals;
            for (float i; iss >> i;) {
                linevals.push_back(i);
            }
            vals.push_back(std::move(linevals));
        }

        auto mat = matrix(vals);

        std::swap(self, mat);

        return is;
    }

    // square matrix dimension.
    size_t dim() const { return dim_; }
    // the logical size in floats not allocated size.
    size_t size() const { return dim() * dim(); }
    // the actual number of allocated bytes including padding. (default impl)
    size_t bytes() const { return row_bytes() * dim(); }

protected:
    size_t dim_;
    std::unique_ptr<float, decltype(&std::free)> data_;

    // padded row size in vector units
    //
    // override for custom padding, all other size functions are implemented in
    // terms of this one.
    constexpr size_t row_sizev() const
    {
        return (dim() + vsize() - 1) / vsize();
    }

    // padded row size in floats
    constexpr size_t row_sizef() const
    {
        auto const rsv = (*this)->row_sizev();
        return rsv * vsize();
    }

    // padded row size in bytes
    constexpr size_t row_bytes() const
    {
        auto const rsv = (*this)->row_sizev();
        return rsv * sizeof(V);
    }

    // Number of floats in the backing vector.
    //
    // the constexpr branch suppresses the compiler warnings when V=float
    static consteval size_t vsize()
    {
        if constexpr (std::same_as<V, float>) {
            return 1;
        }
        else {
            return sizeof(V) / sizeof(float);
        }
    }

private:
    // allow compile-time allocator selection without mucking about with another
    // template parameter.
    //
    // handles alignment internally
    static float* allocate(size_t size)
    {
        if constexpr (std::same_as<V, float>) {
            return static_cast<float*>(std::malloc(size));
        }
        else {
            return static_cast<float*>(_mm_malloc(size, alignof(V)));
        }
    }

    static void free(void* ptr)
    {
        if constexpr (std::same_as<V, float>) {
            std::free(ptr);
        }
        else {
            _mm_free(ptr);
        }
    }
};

template<typename V = m256, size_t H = 6, size_t W = 2>
    requires(vecable<V>)
struct kernel_matrix : public matrix<kernel_matrix<V, H, W>, V> {
    using matrix_t = matrix<kernel_matrix<V, H, W>, V>;
    using matrix<kernel_matrix<V, H, W>, V>::matrix;
    using typename matrix_t::vec_t;

    matrix_t operator*(const matrix_t& rhs) const
    {
        matrix_t result(this->dim());

        constexpr auto vsz = matrix_t::vsize();
        const auto n = this->dim();

        const auto nx = this->pad_heightf();
        const auto ny = this->pad_widthf();

        auto [L1, L2, L3] = cachesize();
        // how many columns of b fit in L3
        const size_t s3 = std::min(L3 / nx / (W * vsz) * (W * vsz), ny);
        // how many rows of a fit in L2
        const size_t s2 = std::min(L2 / ny / H * H, nx);
        // how tall a (k x s3) block in b can be to fit in L1
        const size_t s1 = std::min(L1 / s3, nx);

        // tunable
        // const size_t s3 = 64u;
        // const size_t s2 = 120u;
        // const size_t s1 = 240u;

        const auto& a = *this;
        const auto& b = rhs;
        auto& c = result;

        for (auto i3 = 0u; i3 < ny; i3 += s3) {
            for (auto i2 = 0u; i2 < nx; i2 += s2) {
                for (auto i1 = 0u; i1 < ny; i1 += s1) {

                    for (auto x = i2; x < std::min(i2 + s2, nx); x += H) {
                        for (auto y = i3; y < std::min(i3 + s3, ny);
                             y += W * vsz) {
                            kernel(a, b, c, x, y, i1, std::min(i1 + s1, n), ny);
                        }
                    }
                }
            }
        }
        return result;
    }

    // where h = H and w = W*vsize
    //
    // updates the h*w submatrix C[x:x+h][y:y+w]
    //      using A[x:x+h][l:r] and B[l:r][y:y+w]
    //
    // taking columns of l:r from A and rows l:r from B
    static void kernel(const matrix_t& a, const matrix_t& b, matrix_t& c,
                       size_t x, size_t y, size_t l, size_t r, size_t n)
    {
        // 16(actually 32) registers total in the hardware.
        //
        //  H*W for the kernel accumulator (t)
        //  + W to load b[k,y+j] (W times)
        //  + 1 for the alpha accumulator.
        constexpr auto vsz = matrix_t::vsize();
        // GCC allocates stack space for these even though its unecessary.
        //
        // The specializations for specific kernel sizes are sliiiightly faster
        // and definitely faster for AVX512 where they can use all 32 regs
        vec_t t[H][W]{0};

        // for each column
        for (auto k = l; k < r; ++k) {
            // for each row
            __builtin_prefetch(&b[{k + 1, (y / vsz)}]);
            for (auto i = 0u; i < H; ++i) {
                // load a single float from A and broadcast it into a vector
                vec_t alpha = vec_t{} + a[(x + i), k];

                // The loads of B *should* be hoisted into the outermost loop
                // as they don't depend on i and j is known at compile-time.
                // This uses W registers.
                //
                // emphasis on should.
                //
                // now, for each vector of columns in B
                //  (i.e. `vsize` floats from the current row, at a time)
                for (auto j = 0u; j < W; ++j) {
                    // multiply vsize columns in row k from B
                    // with vsize copies of the value in A[x+i,k]
                    t[i][j] += alpha * b[{k, (y / vsz) + j}];
                }
            }
        }
        // for each vector in the kernel, update the result matrix accordingly
        for (auto i = 0u; i < H; ++i) {
            for (auto j = 0u; j < W; ++j) {
                c[{x + i, (y / vsz) + j}] += t[i][j];
            }
        }
    }

    // specialized 8x2 kernel
    static void kernel(const matrix_t& a, const matrix_t& b, matrix_t& c,
                       size_t x, size_t y, size_t l, size_t r, size_t n)
        requires(std::same_as<V, m512> && H == 8 && W == 2)
    {
        constexpr auto vsz = matrix_t::vsize();

        register vec_t t00 asm("zmm0") = _mm512_setzero_ps();
        register vec_t t01 asm("zmm1") = _mm512_setzero_ps();
        register vec_t t10 asm("zmm2") = _mm512_setzero_ps();
        register vec_t t11 asm("zmm3") = _mm512_setzero_ps();
        register vec_t t20 asm("zmm4") = _mm512_setzero_ps();
        register vec_t t21 asm("zmm5") = _mm512_setzero_ps();
        register vec_t t30 asm("zmm6") = _mm512_setzero_ps();
        register vec_t t31 asm("zmm7") = _mm512_setzero_ps();

        register vec_t t40 asm("zmm8") = _mm512_setzero_ps();
        register vec_t t41 asm("zmm9") = _mm512_setzero_ps();
        register vec_t t50 asm("zmm10") = _mm512_setzero_ps();
        register vec_t t51 asm("zmm11") = _mm512_setzero_ps();
        register vec_t t60 asm("zmm12") = _mm512_setzero_ps();
        register vec_t t61 asm("zmm13") = _mm512_setzero_ps();
        register vec_t t70 asm("zmm14") = _mm512_setzero_ps();
        register vec_t t71 asm("zmm15") = _mm512_setzero_ps();

        for (auto k = l; k < r; ++k) {
            __builtin_prefetch(&b[{k + 1, (y / vsz)}]);
            __builtin_prefetch(&b[{k + 1, (y / vsz) + 1}]);
            register vec_t b0 asm("zmm16") = b[{k, (y / vsz) + 0}];
            register vec_t b1 asm("zmm17") = b[{k, (y / vsz) + 1}];
            vec_t a0 = vec_t{} + a[(x + 0), k];
            t00 += a0 * b0;
            t01 += a0 * b1;
            vec_t a1 = vec_t{} + a[(x + 1), k];
            t10 += a1 * b0;
            t11 += a1 * b1;
            vec_t a2 = vec_t{} + a[(x + 2), k];
            t20 += a2 * b0;
            t21 += a2 * b1;
            vec_t a3 = vec_t{} + a[(x + 3), k];
            t30 += a3 * b0;
            t31 += a3 * b1;
            vec_t a4 = vec_t{} + a[(x + 4), k];
            t40 += a4 * b0;
            t41 += a4 * b1;
            vec_t a5 = vec_t{} + a[(x + 5), k];
            t50 += a5 * b0;
            t51 += a5 * b1;
            vec_t a6 = vec_t{} + a[(x + 6), k];
            t60 += a6 * b0;
            t61 += a6 * b1;
            vec_t a7 = vec_t{} + a[(x + 7), k];
            t70 += a7 * b0;
            t71 += a7 * b1;
        }
        c[{x + 0, (y / vsz) + 0}] += t00;
        c[{x + 0, (y / vsz) + 1}] += t01;
        c[{x + 1, (y / vsz) + 0}] += t10;
        c[{x + 1, (y / vsz) + 1}] += t11;
        c[{x + 2, (y / vsz) + 0}] += t20;
        c[{x + 2, (y / vsz) + 1}] += t21;
        c[{x + 3, (y / vsz) + 0}] += t30;
        c[{x + 3, (y / vsz) + 1}] += t31;
        c[{x + 4, (y / vsz) + 0}] += t40;
        c[{x + 4, (y / vsz) + 1}] += t41;
        c[{x + 5, (y / vsz) + 0}] += t50;
        c[{x + 5, (y / vsz) + 1}] += t51;
        c[{x + 6, (y / vsz) + 0}] += t60;
        c[{x + 6, (y / vsz) + 1}] += t61;
        c[{x + 7, (y / vsz) + 0}] += t70;
        c[{x + 7, (y / vsz) + 1}] += t71;
    }

    // 12x2 kernel
    static void kernel(const matrix_t& a, const matrix_t& b, matrix_t& c,
                       size_t x, size_t y, size_t l, size_t r, size_t n)
        requires(std::same_as<V, m512> && H == 12 && W == 2)
    {
        constexpr auto vsz = matrix_t::vsize();

        register vec_t t00 asm("zmm0") = _mm512_setzero_ps();
        register vec_t t01 asm("zmm1") = _mm512_setzero_ps();
        register vec_t t10 asm("zmm2") = _mm512_setzero_ps();
        register vec_t t11 asm("zmm3") = _mm512_setzero_ps();
        register vec_t t20 asm("zmm4") = _mm512_setzero_ps();
        register vec_t t21 asm("zmm5") = _mm512_setzero_ps();
        register vec_t t30 asm("zmm6") = _mm512_setzero_ps();
        register vec_t t31 asm("zmm7") = _mm512_setzero_ps();

        register vec_t t40 asm("zmm8") = _mm512_setzero_ps();
        register vec_t t41 asm("zmm9") = _mm512_setzero_ps();
        register vec_t t50 asm("zmm10") = _mm512_setzero_ps();
        register vec_t t51 asm("zmm11") = _mm512_setzero_ps();
        register vec_t t60 asm("zmm12") = _mm512_setzero_ps();
        register vec_t t61 asm("zmm13") = _mm512_setzero_ps();
        register vec_t t70 asm("zmm14") = _mm512_setzero_ps();
        register vec_t t71 asm("zmm15") = _mm512_setzero_ps();

        register vec_t t80 asm("zmm16") = _mm512_setzero_ps();
        register vec_t t81 asm("zmm17") = _mm512_setzero_ps();
        register vec_t t90 asm("zmm18") = _mm512_setzero_ps();
        register vec_t t91 asm("zmm19") = _mm512_setzero_ps();
        register vec_t t10_0 asm("zmm20") = _mm512_setzero_ps();
        register vec_t t10_1 asm("zmm21") = _mm512_setzero_ps();
        register vec_t t11_0 asm("zmm22") = _mm512_setzero_ps();
        register vec_t t11_1 asm("zmm23") = _mm512_setzero_ps();

        for (auto k = l; k < r; ++k) {
            __builtin_prefetch(&b[{k + 1, (y / vsz)}]);
            __builtin_prefetch(&b[{k + 1, (y / vsz) + 1}]);
            register vec_t b0 asm("zmm24") = b[{k, (y / vsz) + 0}];
            register vec_t b1 asm("zmm25") = b[{k, (y / vsz) + 1}];

            vec_t a0 = vec_t{} + a[(x + 0), k];
            t00 += a0 * b0;
            t01 += a0 * b1;
            vec_t a1 = vec_t{} + a[(x + 1), k];
            t10 += a1 * b0;
            t11 += a1 * b1;
            vec_t a2 = vec_t{} + a[(x + 2), k];
            t20 += a2 * b0;
            t21 += a2 * b1;
            vec_t a3 = vec_t{} + a[(x + 3), k];
            t30 += a3 * b0;
            t31 += a3 * b1;
            vec_t a4 = vec_t{} + a[(x + 4), k];
            t40 += a4 * b0;
            t41 += a4 * b1;
            vec_t a5 = vec_t{} + a[(x + 5), k];
            t50 += a5 * b0;
            t51 += a5 * b1;
            vec_t a6 = vec_t{} + a[(x + 6), k];
            t60 += a6 * b0;
            t61 += a6 * b1;
            vec_t a7 = vec_t{} + a[(x + 7), k];
            t70 += a7 * b0;
            t71 += a7 * b1;
            vec_t a8 = vec_t{} + a[(x + 8), k];
            t80 += a8 * b0;
            t81 += a8 * b1;
            vec_t a9 = vec_t{} + a[(x + 9), k];
            t90 += a9 * b0;
            t91 += a9 * b1;
            vec_t a10 = vec_t{} + a[(x + 10), k];
            t10_0 += a10 * b0;
            t10_1 += a10 * b1;
            vec_t a11 = vec_t{} + a[(x + 11), k];
            t11_0 += a11 * b0;
            t11_1 += a11 * b1;
        }
        c[{x + 0, (y / vsz) + 0}] += t00;
        c[{x + 0, (y / vsz) + 1}] += t01;
        c[{x + 1, (y / vsz) + 0}] += t10;
        c[{x + 1, (y / vsz) + 1}] += t11;
        c[{x + 2, (y / vsz) + 0}] += t20;
        c[{x + 2, (y / vsz) + 1}] += t21;
        c[{x + 3, (y / vsz) + 0}] += t30;
        c[{x + 3, (y / vsz) + 1}] += t31;
        c[{x + 4, (y / vsz) + 0}] += t40;
        c[{x + 4, (y / vsz) + 1}] += t41;
        c[{x + 5, (y / vsz) + 0}] += t50;
        c[{x + 5, (y / vsz) + 1}] += t51;
        c[{x + 6, (y / vsz) + 0}] += t60;
        c[{x + 6, (y / vsz) + 1}] += t61;
        c[{x + 7, (y / vsz) + 0}] += t70;
        c[{x + 7, (y / vsz) + 1}] += t71;
        c[{x + 8, (y / vsz) + 0}] += t80;
        c[{x + 8, (y / vsz) + 1}] += t81;
        c[{x + 9, (y / vsz) + 0}] += t90;
        c[{x + 9, (y / vsz) + 1}] += t91;
        c[{x + 10, (y / vsz) + 0}] += t10_0;
        c[{x + 10, (y / vsz) + 1}] += t10_1;
        c[{x + 11, (y / vsz) + 0}] += t11_0;
        c[{x + 11, (y / vsz) + 1}] += t11_1;
    }

    static void kernel(const matrix_t& a, const matrix_t& b, matrix_t& c,
                       size_t x, size_t y, size_t l, size_t r, size_t n)
        requires(std::same_as<V, m512> && H == 6 && W == 4)
    {
        constexpr auto vsz = matrix_t::vsize();

        register vec_t t00 asm("zmm0") = _mm512_setzero_ps();
        register vec_t t01 asm("zmm1") = _mm512_setzero_ps();
        register vec_t t02 asm("zmm2") = _mm512_setzero_ps();
        register vec_t t03 asm("zmm3") = _mm512_setzero_ps();

        register vec_t t10 asm("zmm4") = _mm512_setzero_ps();
        register vec_t t11 asm("zmm5") = _mm512_setzero_ps();
        register vec_t t12 asm("zmm6") = _mm512_setzero_ps();
        register vec_t t13 asm("zmm7") = _mm512_setzero_ps();

        register vec_t t20 asm("zmm8") = _mm512_setzero_ps();
        register vec_t t21 asm("zmm9") = _mm512_setzero_ps();
        register vec_t t22 asm("zmm10") = _mm512_setzero_ps();
        register vec_t t23 asm("zmm11") = _mm512_setzero_ps();

        register vec_t t30 asm("zmm12") = _mm512_setzero_ps();
        register vec_t t31 asm("zmm13") = _mm512_setzero_ps();
        register vec_t t32 asm("zmm14") = _mm512_setzero_ps();
        register vec_t t33 asm("zmm15") = _mm512_setzero_ps();

        register vec_t t40 asm("zmm16") = _mm512_setzero_ps();
        register vec_t t41 asm("zmm17") = _mm512_setzero_ps();
        register vec_t t42 asm("zmm18") = _mm512_setzero_ps();
        register vec_t t43 asm("zmm19") = _mm512_setzero_ps();

        register vec_t t50 asm("zmm20") = _mm512_setzero_ps();
        register vec_t t51 asm("zmm21") = _mm512_setzero_ps();
        register vec_t t52 asm("zmm22") = _mm512_setzero_ps();
        register vec_t t53 asm("zmm23") = _mm512_setzero_ps();

        for (auto k = l; k < r; ++k) {
            __builtin_prefetch(&b[{k + 1, (y / vsz)}]);
            __builtin_prefetch(&b[{k + 1, (y / vsz) + 1}]);
            __builtin_prefetch(&b[{k + 1, (y / vsz) + 2}]);
            __builtin_prefetch(&b[{k + 1, (y / vsz) + 3}]);
            register vec_t b0 asm("zmm24") = b[{k, (y / vsz) + 0}];
            register vec_t b1 asm("zmm25") = b[{k, (y / vsz) + 1}];
            register vec_t b2 asm("zmm26") = b[{k, (y / vsz) + 2}];
            register vec_t b3 asm("zmm27") = b[{k, (y / vsz) + 3}];

            vec_t a0 = vec_t{} + a[(x + 0), k];
            t00 += a0 * b0;
            t01 += a0 * b1;
            t02 += a0 * b2;
            t03 += a0 * b3;
            vec_t a1 = vec_t{} + a[(x + 1), k];
            t10 += a1 * b0;
            t11 += a1 * b1;
            t12 += a1 * b2;
            t13 += a1 * b3;
            vec_t a2 = vec_t{} + a[(x + 2), k];
            t20 += a2 * b0;
            t21 += a2 * b1;
            t22 += a2 * b2;
            t23 += a2 * b3;
            vec_t a3 = vec_t{} + a[(x + 3), k];
            t30 += a3 * b0;
            t31 += a3 * b1;
            t32 += a3 * b2;
            t33 += a3 * b3;
            vec_t a4 = vec_t{} + a[(x + 4), k];
            t40 += a4 * b0;
            t41 += a4 * b1;
            t42 += a4 * b2;
            t43 += a4 * b3;
            vec_t a5 = vec_t{} + a[(x + 5), k];
            t50 += a5 * b0;
            t51 += a5 * b1;
            t52 += a5 * b2;
            t53 += a5 * b3;
        }
        c[{x + 0, (y / vsz) + 0}] += t00;
        c[{x + 0, (y / vsz) + 1}] += t01;
        c[{x + 0, (y / vsz) + 2}] += t02;
        c[{x + 0, (y / vsz) + 3}] += t03;

        c[{x + 1, (y / vsz) + 0}] += t10;
        c[{x + 1, (y / vsz) + 1}] += t11;
        c[{x + 1, (y / vsz) + 2}] += t12;
        c[{x + 1, (y / vsz) + 3}] += t13;

        c[{x + 2, (y / vsz) + 0}] += t20;
        c[{x + 2, (y / vsz) + 1}] += t21;
        c[{x + 2, (y / vsz) + 2}] += t22;
        c[{x + 2, (y / vsz) + 3}] += t23;

        c[{x + 3, (y / vsz) + 0}] += t30;
        c[{x + 3, (y / vsz) + 1}] += t31;
        c[{x + 3, (y / vsz) + 2}] += t32;
        c[{x + 3, (y / vsz) + 3}] += t33;

        c[{x + 4, (y / vsz) + 0}] += t40;
        c[{x + 4, (y / vsz) + 1}] += t41;
        c[{x + 4, (y / vsz) + 2}] += t42;
        c[{x + 4, (y / vsz) + 3}] += t43;

        c[{x + 5, (y / vsz) + 0}] += t50;
        c[{x + 5, (y / vsz) + 1}] += t51;
        c[{x + 5, (y / vsz) + 2}] += t52;
        c[{x + 5, (y / vsz) + 3}] += t53;
    }

    static constexpr vec_t set1(float a)
    {
        if constexpr (std::same_as<m128, V>) {
            return _mm_set1_ps(a);
        }
        else if constexpr (std::same_as<m256, V>) {
            return _mm256_set1_ps(a);
        }
        else if constexpr (std::same_as<m512, V>) {
            return _mm512_set1_ps(a);
        }
    }

    static constexpr vec_t set0(void)
    {
        if constexpr (std::same_as<m128, V>) {
            return _mm_setzero_ps();
        }
        else if constexpr (std::same_as<m256, V>) {
            return _mm256_setzero_ps();
        }
        else if constexpr (std::same_as<m512, V>) {
            return _mm512_setzero_ps();
        }
    }

    // padded height and width in floats
    constexpr size_t pad_heightf() const
    {
        return ((this->dim() + H - 1) / H) * H;
    }
    constexpr size_t pad_widthf() const
    {
        // kernel width in floats
        constexpr auto kwf = W * matrix_t::vsize();
        // rounded up to the nearest multiple
        return ((this->dim() + kwf - 1) / kwf) * kwf;
    }

    // padded row size in vector units
    constexpr size_t row_sizev() const // override
    {
        return pad_widthf() / matrix_t::vsize();
    }

    // total allocated bytes including padding
    // (overrides because both height and width are padded)
    constexpr size_t bytes() const // override
    {
        return this->row_sizef() * pad_heightf() * sizeof(float);
    }
};

// implements basic SIMD matrix multiplication
template<typename V = m128>
    requires(vecable<V>)
class simd_matrix : public matrix<simd_matrix<V>, V> {
public:
    using matrix_t = matrix<simd_matrix<V>, V>;
    using matrix<simd_matrix<V>, V>::matrix;
    using typename matrix_t::vec_t;

    matrix_t operator*(const matrix_t& rhs) const
    {
        if (this->dim() != rhs.dim()) {
            throw std::logic_error("mismatched matrix dimensions");
        }
        auto n = this->dim();
        auto rowsz = this->row_sizev();
        auto vecsz = this->vsize();
        auto result = matrix_t(n);

        // transpose rhs matrix for better cache locality
        auto trbm = matrix_t(n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                trbm[i, j] = rhs[j, i];
            }
        }

        const auto& a = *this;
        const auto& b = trbm;
        auto& c = result;

        // c[i,j] = sum(k=0 -> n){ a[i,k] * b[k,j] }
        //
        // NOTE: the indices for b are transposed in the implementation
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                // zero-init accumulator
                vec_t acc{};

                // vertical sum
                for (size_t k = 0; k < rowsz; ++k)
                    acc += a[{i, k}] * b[{j, k}];

                // horizontal sum
                for (size_t k = 0; k < vecsz; ++k)
                    c[i, j] += acc[k];
            }
        }
        return result;
    }
};

// implements naive matrix multiplication.
class naive_matrix : public matrix<naive_matrix, float> {
public:
    using matrix_t = matrix<naive_matrix, float>;
    using matrix_t::matrix;
    using typename matrix_t::vec_t;

    matrix_t operator*(const matrix_t& rhs) const
    {
        if (this->dim() != rhs.dim()) {
            throw std::logic_error("mismatched matrix dimensions");
        }
        // signed comparison makes loops faster because compiler can assume
        // overflow doesn't occur. (undefined behaviour)
        ptrdiff_t n = this->dim();
        auto result = matrix_t(n);

        const auto& a = *this;
        const auto& b = rhs;
        auto& c = result;

        // c[i,j] = sum(k=0 -> n){ a[i,k] * b[k,j] }
        for (auto i = 0l; i < n; ++i) {
            for (auto j = 0l; j < n; ++j) {
                for (auto k = 0l; k < n; ++k) {
                    c[i, j] += a[i, k] * b[k, j];
                }
            }
        }
        return result;
    }
};

#if defined(__AVX512F__)
using basic_matrix = kernel_matrix<m256, 8, 2>;
#elif defined(__AVX2__)
using basic_matrix = kernel_matrix<m256, 8, 2>;
#else
using basic_matrix = kernel_matrix<m128, 8, 2>;
#endif

#endif
