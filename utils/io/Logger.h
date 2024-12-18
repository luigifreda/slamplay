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
#pragma once

#include <pthread.h>
#include <fstream>
#include <iostream>
#include "io/Log.h"
#include "io/LogColors.h"

namespace slamplay {

///	\class Logger
///	\author Luigi Freda
///	\brief A class implementing a logger which is capable of intercepting std::endl
///	\note
/// 	\todo
///	\date
///	\warning
class Logger {
   public:
    Logger(const std::string &aname, const std::string &color = LOG_COL_NORMAL, std::ostream &out = std::cout) : _ofile(out), _name(aname), _color(color), _bFirst(true) {}

    template <typename T>
    Logger &operator<<(const T &a) {
        if (_bFirst)
        {
            _ofile << _color << "[" << _name << "]: " << LOG_COL_NORMAL;
            _bFirst = false;
        }  // put name at first
        _ofile << a;
        return *this;
    }

    Logger &operator<<(std::ostream &(*pf)(std::ostream &)) {
        // here we intercept std::endl
        _ofile << pf;
        _bFirst = false;  // reset first-flag at the end of line
        return *this;
    }

   protected:
    std::ostream &_ofile;
    std::string _name;
    std::string _color;
    bool _bFirst;
};

///	\class LoggerFile
///	\author Luigi Freda
///	\brief A class implementing a logger which writes on a file (it is capable of intercepting std::endl)
///	\note
/// 	\todo
///	\date
///	\warning

class LoggerFile {
   public:
    LoggerFile(const std::string &filename) : _filename(filename) {
        if (!filename.empty())
        {
            _ofile.open(filename.c_str(), std::fstream::out);
            if (!_ofile.is_open())
            {
                LogError << "LoggerFile: unable to open " << filename;
            }
        } else
        {
            LogError << "LoggerFile: filename empty";
        }
    }

    ~LoggerFile() {
        if (_ofile.is_open())
        {
            _ofile.close();
        }
    }

    template <typename T>
    LoggerFile &operator<<(const T &a) {
        _ofile << a;
        return *this;
    }

    LoggerFile &operator<<(std::ostream &(*pf)(std::ostream &)) {
        // here we intercept std::endl
        _ofile << pf;
        return *this;
    }

    /// Writes the block of data pointed by s, with a size of n characters, into the output buffer
    void Write(const char *s, std::streamsize n) {
        _ofile.write(s, n);
    }

    void Clear() {
        _ofile.clear();
    }

   protected:
    std::fstream _ofile;
    std::string _filename;
};

}  // namespace slamplay