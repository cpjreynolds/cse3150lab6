#ifndef CPUINFO_HPP
#define CPUINFO_HPP

#include <tuple>
#include <cstddef>

// {L1, L2, L3} data cache sizes in bytes.
std::tuple<int, int, int> cachesize();

// cpu base frequency in Hz
size_t cpu_frequency();

#endif
