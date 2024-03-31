#include "matrix.hpp"

#include <fstream>
#include <iostream>

using std::ifstream;

static basic_matrix exact0paths(basic_matrix& a, basic_matrix& b,
                                basic_matrix& c)
{
    basic_matrix* mat[3] = {&a, &b, &c};
    auto n = a.dim();
    basic_matrix unified(n);

    // do initialization
    for (auto i = 0u; i < n; ++i) {
        for (auto j = 0u; j < n; ++j) {
            for (auto k = 0u; k < 3; ++k) {
                auto elt = (*mat[k])[i, j];
                if (elt == 1.0) {
                    unified[i, j] = 3 * (n + 1);
                }
                else if (elt == -1.0) {
                    unified[i, j] = 1.0 / (3 * (n + 1));
                }
                else {
                    // do nothing because the unified matrix is already
                    // initialized to 0.0
                }
            }
        }
    }

    // compute zero paths and accumulate in result matrix.
    basic_matrix result(n);
    auto acc = unified * unified;
    for (auto x = 2u; x < (3 * n * n + 1); ++x) {
        for (auto i = 0u; i < n; ++i) {
            for (auto j = 0u; j < n; ++j) {
                if (cmpf(acc[i, j], 1.0)) {
                    // we found a zero path
                    result[i, j] = 1.0;
                }
            }
        }
        acc = acc * unified;
    }

    return result;
}

#ifndef TESTING

// does the thing the assignment page requires
static basic_matrix do_3file_input(const char* argv[])
{
    std::ifstream ifiles[] = {ifstream(argv[0]), ifstream(argv[1]),
                              ifstream(argv[2])};

    for (int i = 0; i < 3; ++i) {
        if (!ifiles[i].is_open()) {
            std::string msg(argv[i + 1]);
            throw std::runtime_error(msg + ": no such file");
        }
    }

    basic_matrix mats[3];
    for (int i = 0; i < 3; ++i) {
        ifiles[i] >> mats[i];
    }

    return exact0paths(mats[0], mats[1], mats[2]);
}

int main(int argc, const char* argv[])
{
    const char* default_fnames[3] = {"tmat-1.txt", "tmat0.txt", "tmat1.txt"};

    basic_matrix result;

    if (argc != 4) {
        result = do_3file_input(default_fnames);
    }
    else {
        result = do_3file_input(argv + 1);
    }

    std::cout << "zero cost paths represented with 1.0:\n";
    std::cout << result << std::endl;

    return EXIT_SUCCESS;
}

#else
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("exact0paths")
{
    basic_matrix allones(4, 1.0);

    basic_matrix tm1{{-1, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}};
    basic_matrix t0{{2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}};
    basic_matrix t1{{2, 1, 2, 2}, {2, 2, 1, 2}, {2, 2, 2, 1}, {1, 2, 2, 2}};

    CHECK(exact0paths(tm1, t0, t1) == allones);
}
#endif
