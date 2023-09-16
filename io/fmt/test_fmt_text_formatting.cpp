#include "fmt/core.h"
#if GCC_VERSION >= 10
#include <numbers>
#endif 
#include <iostream>

int main(int argc, char **argv) 
{
  const auto theAnswer = fmt::format("The answer is {}.", 42);
  std::cout << theAnswer << "\n";

  // Many different format specifiers are possible. 
  const auto formattedNumbers =
    fmt::format("Decimal: {:f}, Scientific: {:e}, Hexadecimal: {:X}",
      3.1415, 0.123, 255);
  std::cout << formattedNumbers << "\n";

  // Arguments can be reordered in the created string by using an index {n:}:
  const auto reorderedArguments =
    fmt::format("Decimal: {1:f}, Scientific: {2:e}, Hexadecimal: {0:X}",
      255, 3.1415, 0.123);
  std::cout << reorderedArguments << "\n";

#if GCC_VERSION >= 10
  // The number of decimal places can be specified as follows:
  const auto piWith22DecimalPlaces = fmt::format("PI = {:.22f}",
    std::numbers::pi);
  std::cout << piWith22DecimalPlaces << "\n";
#endif 
  return 0;
}