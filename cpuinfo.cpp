#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <tuple>
#include <algorithm>

#include <cpuid.h>
#include "cpuinfo.hpp"

/*
 * How in the happy hell is there not an easier way of doing this portably.
 *
 * -__-
 *
 */

std::tuple<int, int, int> cachesize()
{
    // defaults
    int l1, l2, l3;
    l1 = l2 = l3 = -1;

    if (__builtin_cpu_is("intel")) {
        // "Intel Processor Identification and the CPUID Instruction"
        //      Application Note 485 (May 2012)
        //      Section 5.1.5
        //
        if (__get_cpuid_max(0, nullptr) < 4)
            // leaf 4 isn't available.
            return {l1, l2, l3};

        uint32_t eax, ebx, ecx, edx;
        // iterate over cpuid leaf 4
        for (int i = 0; __get_cpuid_count(4, i, &eax, &ebx, &ecx, &edx); ++i) {
            // eax[4:0] but only [3:0] matter
            uint32_t type = eax & 0xf;
            if (type == 0)
                break;
            // Cache Level (starts at 1)
            // eax[7:5]
            uint32_t level = (eax >> 5) & 0x7;
            // System Coherency Line Size
            // ebx[11:0] + 1
            uint32_t linesz = (ebx & 0xfff) + 1;
            // Physical Line partitions
            // ebx[21:12] + 1
            uint32_t parts = ((ebx >> 12) & 0x3ff) + 1;
            // Ways of Associativity
            // ebx[31:22] + 1
            uint32_t ways = ((ebx >> 22) & 0x3ff) + 1;
            // Number of Sets
            // ecx[31:0] + 1
            uint32_t sets = ecx + 1;
            // Size in bytes
            uint32_t size = ways * parts * linesz * sets;
            switch (level) {
            case 1:
                if (type == 1) { // is data cache
                    l1 = size;
                }
                break;
            case 2:
                l2 = size;
                break;
            case 3:
                l3 = size;
                break;
            }
        }
    }
    else if (__builtin_cpu_is("amd")) {
        uint32_t eax, ebx, ecx, edx;
        if (__get_cpuid(0x80000005, &eax, &ebx, &ecx, &edx)) {
            l1 = ((ecx >> 24) & 0xFF) * 1024;
        }
        if (__get_cpuid(0x80000006, &eax, &ebx, &ecx, &edx)) {
            l2 = ((ecx >> 16) & 0xffff) * 1024;
        }
        if (__get_cpuid(0x8000001d, &eax, &ebx, &ecx, &edx)) {
            for (int i = 0;
                 __get_cpuid_count(0x8000001d, i, &eax, &ebx, &ecx, &edx);
                 ++i) {
                uint32_t type = eax & 0xf;
                if (type == 0)
                    break;
                uint32_t level = (eax >> 5) & 0x7;
                uint32_t sets = ecx + 1;
                uint32_t linesz = (ebx & 0x7ff) + 1;
                uint32_t parts = ((ebx >> 12) & 0x1ff) + 1;
                uint32_t ways = ((ebx >> 22) & 0x1ff) + 1;
                uint32_t size = sets * linesz * parts * ways;

                switch (level) {
                case 1:
                    if (type == 1)
                        l1 = size;
                    break;
                case 2:
                    l2 = size;
                    break;
                case 3:
                    l3 = size;
                    break;
                }
            }
        }
    }

    return {l1, l2, l3};
}

static const char* brand_string_impl()
{
    static char bstr[48];

    if (__get_cpuid_max(0x80000000, nullptr) < 0x80000004) {
        std::memcpy(bstr, "unknown", sizeof("unknown"));
    }
    else {
        uint32_t eax, ebx, ecx, edx;
        eax = ebx = ecx = edx = 0;

        auto bstri = reinterpret_cast<uint32_t*>(&bstr);

        for (auto i = 0u; i < 3; ++i) {
            __get_cpuid(0x80000002 + i, &eax, &ebx, &ecx, &edx);
            bstri[4 * i + 0] = eax;
            bstri[4 * i + 1] = ebx;
            bstri[4 * i + 2] = ecx;
            bstri[4 * i + 3] = edx;
        }
    }
    return bstr;
}

std::string_view brand_string()
{
    // only runs once
    static const std::string_view bstr{brand_string_impl()};
    return bstr;
}

// yes, this is actually the OEM recommended, portable method of getting a CPU
// base frequency.
//
// may the machine gods forgive us.
//
// See: Intel® 64 and IA-32 Architectures Software Developer’s Manual
//          Figure 3-10. Algorithm for Extracting Processor Frequency
size_t cpu_frequency_impl()
{
    auto bstr = brand_string();

    auto hzidx = bstr.rfind("Hz");
    if (hzidx == bstr.npos)
        return 0;
    bstr.remove_suffix(bstr.size() - hzidx);

    size_t mult = 0;
    switch (bstr.back()) {
    case 'M':
        mult = 1e6;
        break;
    case 'G':
        mult = 1e9;
        break;
    case 'T':
        mult = 1e12;
        break;
    default:
        return 0;
    }
    bstr.remove_suffix(1);

    auto rb = bstr.rbegin();
    auto re = bstr.rend();

    auto endit = std::find_if(rb, re, [](auto c) { return std::isblank(c); });
    auto endidx = std::distance(endit, re);
    auto freqstr = bstr.substr(endidx);

    // simply converting to float and multiplying would be inaccurate as not all
    // frequencies in Hz will be exactly representable as float32.
    size_t dotpos;
    if ((dotpos = freqstr.find('.')) != freqstr.npos) {
        auto lhs = freqstr.data();
        auto lhe = freqstr.data() + dotpos;
        auto rhs = lhe + 1;
        auto rhe = freqstr.data() + freqstr.size();

        size_t lhv;
        if (std::from_chars(lhs, lhe, lhv).ec != std::errc()) {
            return 0;
        }

        size_t rhv = 0;
        for (auto it = rhs; it != rhe; ++it) {
            size_t val = 0;
            if (std::from_chars(it, it + 1, val).ec != std::errc()) {
                return 0;
            }
            rhv += val * (mult / (1e1 * (it - rhs + 1)));
        }
        return (lhv * mult) + rhv;
    }
    else {
        size_t val;
        if (std::from_chars(freqstr.data(), freqstr.data() + freqstr.size(),
                            val)
                .ec != std::errc()) {
            return 0;
        }
        return val * mult;
    }
}

size_t cpu_frequency()
{
    static const size_t freq = cpu_frequency_impl();
    return freq;
}
