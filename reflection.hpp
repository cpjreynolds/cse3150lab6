// little utility to return the name of a type as a constexpr string.
#ifndef REFLECTION_HPP
#define REFLECTION_HPP

#include <source_location>
#include <string_view>

template<typename T>
static consteval auto func_name()
{
    const auto& loc = std::source_location::current();
    return loc.function_name();
}

template<typename T>
static consteval std::string_view type_of_impl()
{
    constexpr std::string_view fname = func_name<T>();
    return {fname.begin() + fname.find('=') + 2, fname.end() - 1};
}

template<typename T>
constexpr std::string_view type_of(T&& arg)
{
    return type_of_impl<decltype(arg)>();
}

template<typename T>
constexpr std::string_view type_of()
{
    return type_of_impl<T>();
}

#endif
