/*
 * Graders beware. Here be dragons.
 *
 * (I had a lot of fun)
 */
#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <memory>
#include <new>
#include <initializer_list>
#include <random>
#include <utility>
#include <algorithm>
#include <version>

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
concept vecable = std::same_as<T, float> || std::same_as<T, m128>
#ifdef __AVX2__
                  || std::same_as<T, m256>
#endif
#ifdef __AVX512F__
                  || std::same_as<T, m512>
#endif
    ;

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
    requires vecable<V>
class matrix {
public:
    using vec_t = vec_t_helper<V>::type;

    explicit matrix(size_t n, float init = 0.0)
        : dim_{n}, data_{matrix::allocate((*this)->bytes()), &matrix::free}
    {
        // I actually needed this during testing. large alignment
        // values cause aligned_alloc to return null.
        //
        // using _mm_malloc() instead for now.
        if (data() == nullptr) {
            throw std::bad_alloc();
        }
        std::memset(data(), init, (*this)->bytes());
    };

    T* operator->() { return static_cast<T*>(this); }

    matrix(std::initializer_list<std::initializer_list<float>> init)
        : matrix(init.size())
    {
        if (init.size() != init.begin()->size()) {
            throw std::logic_error("mismatched matrix dimensions");
        }
        size_t i = 0;
        for (auto row : init) {
            std::memcpy((*this)[i], row.begin(), dim());
            ++i;
        }
    }

    template<std::input_iterator I>
        requires std::same_as<std::iter_value_t<I>, float>
    matrix(size_t n, I fst, I lst) : matrix(n)
    {
        if (std::distance(fst, lst) != n * n) {
            throw std::logic_error("mismatched matrix dimensions");
        }
        for (auto i = 0u; i < n; ++i) {
            for (auto j = 0u; j < n; ++j) {
                (*this)[i, j] = fst++;
            }
        }
    }

    // return a matrix filled with random floats
    static matrix random(size_t n)
    {
        matrix mat(n);
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.0, 1.0);
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

    float* operator[](size_t r) { return data() + (row_sizef() * r); }

    const float* operator[](size_t r) const
    {
        return data() + (row_sizef() * r);
    }

    // woohoo c++23 multidimensional indexing
    float& operator[](size_t r, size_t c)
    {
        return data()[(row_sizef() * r) + c];
    }

    const float& operator[](size_t r, size_t c) const
    {
        return data()[(row_sizef() * r) + c];
    }

    vec_t& operator[](std::pair<size_t, size_t> idx)
    {
        // very legal and very cool
        vec_t* d = reinterpret_cast<vec_t*>(data());
        return d[row_sizev() * idx.first + idx.second];
    }

    const vec_t& operator[](std::pair<size_t, size_t> idx) const
    {
        // undefined behaviour? what undefined behaviour?
        const vec_t* d = reinterpret_cast<const vec_t*>(data());
        return d[row_sizev() * idx.first + idx.second];
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

    // square matrix dimension.
    size_t dim() const { return dim_; }
    // the logical size, not allocated size.
    size_t size() const { return dim() * dim(); }
    // the actual number of allocated bytes including padding. (default impl)
    size_t bytes() const { return row_bytes() * dim(); }

    friend std::ostream& operator<<(std::ostream& os, const matrix& self)
    {
        for (size_t i = 0; i < self.dim(); ++i) {
            for (size_t j = 0; j < self.dim(); ++j) {
                os << self[i, j] << ' ';
            }
            os << '\n';
        }
        return os;
    }

protected:
    size_t dim_;
    std::unique_ptr<float, decltype(&std::free)> data_;

    // padded row size in vector units
    constexpr size_t row_sizev() const
    {
        return (dim() + vsize() - 1) / vsize();
    }

    // padded row size in floats
    constexpr size_t row_sizef() const { return row_sizev() * vsize(); }

    // padded row size in bytes
    constexpr size_t row_bytes() const { return row_sizev() * sizeof(V); }

    // size of the backing vectors
    //
    // the constexpr branch suppresses the compiler warnings when V=float
    static constexpr size_t vsize()
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

template<typename V = m128, size_t H = 6, size_t W = 16>
    requires(vecable<V> && !std::same_as<V, float>)
struct kernel_matrix : public matrix<kernel_matrix<V, H, W>, V> {
    using matrix_t = matrix<kernel_matrix<V, H, W>, V>;
    // using matrix_t::matrix;
    using typename matrix_t::vec_t;

    matrix_t operator*(const matrix_t& rhs) const
    {
        matrix_t result(this->dim());

        const auto n = this->dim();
        const float* a = this->data();
        const vec_t* b = rhs.vdata();
        vec_t* c = result.vdata();

        const auto nx = this->pad_heightf();
        const auto ny = this->pad_widthf();

        auto [l1, l2, l3] = cachesize();
        const size_t s3 = std::min(l3 / nx / W * W, ny);
        const size_t s2 = std::min(l2 / ny / H * H, nx);
        const size_t s1 = std::min(l1 / s3, nx);

        for (auto i3 = 0u; i3 < ny; i3 += s3) {
            for (auto i2 = 0u; i2 < nx; i2 += s2) {
                for (auto i1 = 0u; i1 < ny; i1 += s1) {

                    for (auto x = i2; x < std::min(i2 + s2, nx); x += H) {
                        for (auto y = i3; y < std::min(i3 + s3, ny); y += W) {
                            kernel(a, b, c, x, y, i1, std::min(i1 + s1, n), ny);
                        }
                    }
                }
            }
        }
        return result;
    }

    static void kernel(const float* a, const vec_t* b, vec_t* __restrict__ c,
                       size_t x, size_t y, size_t l, size_t r, size_t n)
    {
        constexpr size_t hv = H;
        constexpr size_t wv = W / matrix_t::vsize();
        constexpr size_t vsz = matrix_t::vsize();
        static_assert(W % matrix_t::vsize() == 0, "bad kernel width");
        // vec_t t[hv][wv] = {set0()};

        if constexpr (std::same_as<m256, V>) {
            register vec_t t0 asm("ymm0") = set0();
            register vec_t t1 asm("ymm1") = set0();
            register vec_t t2 asm("ymm2") = set0();
            register vec_t t3 asm("ymm3") = set0();
            register vec_t t4 asm("ymm4") = set0();
            register vec_t t5 asm("ymm5") = set0();
            register vec_t t6 asm("ymm6") = set0();
            register vec_t t7 asm("ymm7") = set0();
            register vec_t t8 asm("ymm8") = set0();
            register vec_t t9 asm("ymm9") = set0();
            register vec_t t10 asm("ymm10") = set0();
            register vec_t t11 asm("ymm11") = set0();
            vec_t t[6][2] = {{t0, t1}, {t2, t3}, {t4, t5},
                             {t6, t7}, {t8, t9}, {t10, t11}};
            for (auto k = l; k < r; ++k) {
                for (auto i = 0u; i < hv; ++i) {
                    vec_t alpha = set1(a[(x + i) * n + k]);
                    for (auto j = 0u; j < wv; ++j) {
                        t[i][j] += alpha * b[(k * n + y) / vsz + j];
                    }
                }
            }
            for (auto i = 0u; i < hv; ++i) {
                for (auto j = 0u; j < wv; ++j) {
                    c[((x + i) * n + y) / vsz + j] += t[i][j];
                }
            }
        }
        else if constexpr (std::same_as<m512, V>) {
            register vec_t t0 asm("zmm0") = set0();
            register vec_t t1 asm("zmm1") = set0();
            register vec_t t2 asm("zmm2") = set0();
            register vec_t t3 asm("zmm3") = set0();
            register vec_t t4 asm("zmm4") = set0();
            register vec_t t5 asm("zmm5") = set0();
            vec_t t[6][1] = {{t0}, {t1}, {t2}, {t3}, {t4}, {t5}};
            for (auto k = l; k < r; ++k) {
                for (auto i = 0u; i < hv; ++i) {
                    vec_t alpha = set1(a[(x + i) * n + k]);
                    for (auto j = 0u; j < wv; ++j) {
                        t[i][j] += alpha * b[(k * n + y) / vsz + j];
                    }
                }
            }
            for (auto i = 0u; i < hv; ++i) {
                for (auto j = 0u; j < wv; ++j) {
                    c[((x + i) * n + y) / vsz + j] += t[i][j];
                }
            }
        }
        else {
            vec_t t[hv][wv] = {set0()};
            for (auto k = l; k < r; ++k) {
                for (auto i = 0u; i < hv; ++i) {
                    vec_t alpha = set1(a[(x + i) * n + k]);
                    for (auto j = 0u; j < wv; ++j) {
                        t[i][j] += alpha * b[(k * n + y) / vsz + j];
                    }
                }
            }
            for (auto i = 0u; i < hv; ++i) {
                for (auto j = 0u; j < wv; ++j) {
                    c[((x + i) * n + y) / vsz + j] += t[i][j];
                }
            }
        }
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
        return ((this->dim() + W - 1) / W) * W;
    }

    // overrides because of different padding requirements
    size_t bytes() const
    {
        return this->row_sizef() * pad_heightf() * this->vsize();
    }
};

// implements basic SIMD matrix multiplication
template<typename V = m128>
class simd_matrix : public matrix<simd_matrix<V>, V> {
public:
    using matrix_t = matrix<simd_matrix<V>, V>;
    using matrix_t::matrix;
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
        auto _b = rhs.data();
        vec_t* b = reinterpret_cast<vec_t*>(trbm.data());
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                b[i * rowsz + j / vecsz][j % vecsz] = _b[j * n + i];
                // trbm[i, j] = rhs[j, i];
            }
        }

        const vec_t* a = reinterpret_cast<const vec_t*>(this->data());
        float* c = result.data();

        /*
        const auto& a = *this;
        const auto& b = trbm;
        auto& c = result;
        */

        // c[i,j] = sum(k=0 -> n){ a[i,k] * b[k,j] }
        //
        // NOTE: the indices for b are transposed in the implementation
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                // zero-init accumulator
                vec_t acc{};

                // vertical sum
                for (size_t k = 0; k < rowsz; ++k)
                    acc += a[i * rowsz + k] * b[j * rowsz + k];
                // acc += a[{i, k}] * b[{j, k}];

                // horizontal sum
                for (size_t k = 0; k < vecsz; ++k)
                    c[i * n + j] += acc[k];
                // c[i, j] += acc[k];
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

#endif
