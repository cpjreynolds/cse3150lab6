#ifndef INSTRUMENT_HPP
#define INSTRUMENT_HPP

#include <cstdlib>
#include <cstddef>
#include <ctime>
#include <initializer_list>
#include <iomanip>
#include <functional>
#include <iostream>
#include <string_view>
#include <utility>
#include <map>

#include "reflection.hpp"

// creates a read-write memory barrier in the compiler.
//
// effectively prevents it from optimizing away a result.
template<typename T>
inline void barrier(T&& val)
{
    asm volatile("" : "+m"(val) : : "memory");
}

// parameters:
//      M   => matrix type
//      n   => matrix dimension
//      t   => test duration
//
// returns: {seconds per iteration, number of iterations}
//
// creates random matrices to multiply together each iteration, but only times
// the multiplication operation.
template<typename M>
inline std::pair<float, size_t> time_mult(size_t dim, std::clock_t t)
{
    std::clock_t total{};

    std::clock_t maxtime = t * CLOCKS_PER_SEC;

    size_t ctr = 0;

    while (total < maxtime) {
        auto m1 = M::random(dim);
        auto m2 = M::random(dim);

        auto start = std::clock();
        auto m3 = m1 * m2;
        total += (std::clock() - start);
        ++ctr;

        barrier(m3);
    }
    float spi = float(total) / CLOCKS_PER_SEC / ctr;
    return {spi, ctr};
}

// class to hold a suite of benchmarks and defer execution until ready
struct benchmarks {
    using bench_fn =
        std::function<std::pair<float, size_t>(size_t, std::clock_t)>;

    template<typename... Ms>
    void add(size_t n, std::clock_t secs = 1)
    {
        using std::pair, std::string_view;
        using benchlist = std::initializer_list<pair<string_view, bench_fn>>;

        benchlist benches = {{type_of<Ms>(), time_mult<Ms>}...};

        for (auto [mname, bench] : benches) {

            namewidth = std::max(namewidth, mname.size());
            sizewidth = std::max(std::to_string(n).size() * 2 + 1, sizewidth);

            benchmap.insert({{n, std::string(mname), secs}, bench});
        }
    }

    void run()
    {
        for (auto& [data, bench] : benchmap) {
            auto [n, mname, secs] = data;
            auto [spi, ctr] = bench(n, secs);

            float gflops = (float(n) * n * n) / (spi * 1e9);

            std::ostringstream msize;
            msize << n << 'x' << n;
            std::ostringstream flopstream;
            flopstream << std::setprecision(1) << std::fixed << gflops;
            auto flopstr = flopstream.str();
            if (flopstr.starts_with('0')) {
                flopstr.erase(0, 1);
            }
            std::cout << std::setw(namewidth) << mname << ": "
                      << std::setw(sizewidth) << msize.str() << " | ";
            std::cout << std::setw(5) << flopstr << " gflops" << std::endl;
        }
    }

private:
    std::map<std::tuple<size_t, std::string, std::clock_t>, bench_fn> benchmap;
    size_t sizewidth = 0;
    size_t namewidth = 0;
};

#endif
