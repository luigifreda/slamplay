#pragma once

#include <pthread.h>
#include <iostream>
#include <fstream>
#include "LogColors.h"
#include "Log.h"

///	\class Logger
///	\author Luigi Freda
///	\brief A class implementing a logger which is capable of intercepting std::endl
///	\note
/// 	\todo 
///	\date
///	\warning
class Logger
{
public:

    Logger(const std::string &aname, const std::string& color = LOG_COL_NORMAL, std::ostream &out = std::cout) : _ofile(out), _name(aname), _color(color), _bFirst(true)
    {}

    template <typename T>
    Logger &operator<<(const T &a)
    {
        if (_bFirst)
        {
            _ofile << _color << "[" << _name << "]: " << LOG_COL_NORMAL;
            _bFirst = false;
        } // put name at first
        _ofile << a;
        return *this;
    }

    Logger &operator<<(std::ostream& (*pf) (std::ostream&))
    {
        // here we intercept std::endl
        _ofile << pf;
        _bFirst = false; // reset first-flag at the end of line 
        return *this;
    }

protected:
    std::ostream& _ofile;
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

class LoggerFile
{
public:

    LoggerFile(const std::string &filename) : _filename(filename)
    {
        if (!filename.empty())
        {
            _ofile.open(filename.c_str(), std::fstream::out);
            if (!_ofile.is_open())
            {
                LogError << "LoggerFile: unable to open " << filename;
            }
        }
        else
        {
            LogError << "LoggerFile: filename empty";
        }
    }

    ~LoggerFile()
    {
        if (_ofile.is_open())
        {
            _ofile.close();
        }
    }

    template <typename T>
    LoggerFile &operator<<(const T &a)
    {
        _ofile << a;
        return *this;
    }

    LoggerFile &operator<<(std::ostream& (*pf) (std::ostream&))
    {
        // here we intercept std::endl
        _ofile << pf;
        return *this;
    }

    /// Writes the block of data pointed by s, with a size of n characters, into the output buffer
    void Write(const char* s, std::streamsize n)
    {
        _ofile.write(s, n);
    }

    void Clear()
    {
        _ofile.clear();
    }

protected:
    std::fstream _ofile;
    std::string _filename;
};
