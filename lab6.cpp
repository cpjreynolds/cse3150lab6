#include "matrix.hpp"
#ifndef TESTING

int main(int argc, char** argv)
{
    auto a = kernel_matrix<m512, 8, 2>::random(64);
    auto b = kernel_matrix<m512, 8, 2>::random(64);

    auto c = a * b;

    std::cout << c << std::endl;

    return EXIT_SUCCESS;
}

#else
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#endif
