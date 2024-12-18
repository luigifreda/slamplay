// *************************************************************************
/* 
 * This file is part of the slamplay project.
 * Copyright (C) 2018-present Luigi Freda <luigifreda at gmail dot com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version, at your option. If this file is a modified/adapted 
 * version of an original file distributed under a different license that 
 * is not compatible with the GNU General Public License, the 
 * BSD 3-Clause License will apply instead.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
// *************************************************************************
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