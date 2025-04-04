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
#include <iostream>
#include <string>

#include <rerun.hpp>

/*
Launch this with
echo 'hello from stdin!' | ./build/test_rerun_stdio | rerun -

*/
int main() {
    const auto rec = rerun::RecordingStream("rerun_example_stdio");
    rec.to_stdout().exit_on_failure();

    std::string input;
    std::string line;
    while (std::getline(std::cin, line)) {
        input += line + '\n';
    }

    rec.log("stdin", rerun::TextDocument(input));

    return 0;
}
