#include <cstdlib>

#ifndef TESTING

int main(int argc, char** argv)
{
    return EXIT_SUCCESS;
}

#else
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#endif
