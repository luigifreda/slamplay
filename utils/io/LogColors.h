#pragma once

#include <iostream>


#define LOG_COL_NORMAL "\033[0m"
#define LOG_COL_BLACK "\033[22;30m"
#define LOG_COL_RED "\033[22;31m"
#define LOG_COL_GREEN "\033[22;32m"
#define LOG_COL_BROWN "\033[22;33m"
#define LOG_COL_BLUE "\033[22;34m"
#define LOG_COL_MAGENTA "\033[22;35m"
#define LOG_COL_CYAN "\033[22;36m"
#define LOG_COL_GRAY "\033[22;37m"
#define LOG_COL_DARK_GRAY "\033[01;30m"
#define LOG_COL_LIGHT_RED "\033[01;31m"
#define LOG_COL_LIGHT_GREEN "\033[01;32m"
#define LOG_COL_YELLOW "\033[01;33m"
#define LOG_COL_LIGHT_BLUE "\033[01;34m"
#define LOG_COL_LIGHT_MAGENTA "\033[01;35m"
#define LOG_COL_LIGH_CYAN "\033[01;36m"
#define LOG_COL_WHITE "\033[01;37m"


#define STDIO_COL_NORMAL  "\x1B[0m"
#define STDIO_COL_RED  "\x1B[31m"
#define STDIO_COL_GREEN  "\x1B[32m"
#define STDIO_COL_YELLOW  "\x1B[33m"
#define STDIO_COL_BLU  "\x1B[34m"
#define STDIO_COL_MAGENTA  "\x1B[35m"
#define STDIO_COL_CYAN  "\x1B[36m"
#define STDIO_COL_WHITE  "\x1B[37m"
/// usage printf("%sred\n", STDIO_COLOR_X);



///\class LogColor 
///\brief Basic class for log color management (similar to GLColor)
///\author 
class LogColor
{
	/// \brief Default constructor
	public: LogColor(): r(0), g(0), b(0), a(1) {}
	
	/// \brief Constructor
	public: LogColor(float _r, float _g, float _b, float _a = 1.) : r(_r), g(_g), b(_b), a(_a) {}

	/// Red color information
	public: float r;
	
	/// Green color information
	public: float g;
	
	/// Blue color information
	public: float b;
	
	/// Alpha color information
	public: float a;
};

inline std::ostream &operator<<(std::ostream &output, const LogColor &color) 
{
    output << "("<<color.r<<","<<color.g<<","<<color.b<<","<<color.a<<")";
    return output;
}

// inverse of the previous output 
inline std::istream &operator>>(std::istream &input, LogColor &color) 
{
    char c; 
    input >> c >> color.r >> c >> color.g >> c >> color.b >> c >> color.a >> c;
    return input;
}


///\namespace LogColors
///\brief Some predefined colors.
///\author 
namespace LogColors
{

	static const float NONE		= 0.00;
	static const float DARK		= 0.33;
	static const float MEDIUM	= 0.66;
	static const float LIGHT	= 0.99;


	inline const LogColor Black()		{return LogColor(NONE,NONE,NONE);}
	inline const LogColor White()		{return LogColor(1.,1.,1.);}
	/*grey*/
	inline const LogColor DarkGrey()		{return LogColor(DARK,DARK,DARK);}
	inline const LogColor LightGrey()	{return LogColor(MEDIUM,MEDIUM,MEDIUM);}
	/*red*/
	inline const LogColor LightRed()		{return LogColor(LIGHT,NONE,NONE);}
	inline const LogColor Red()		{return LogColor(MEDIUM,NONE,NONE);}
	inline const LogColor DarkRed()		{return LogColor(DARK,NONE,NONE);}
	/*green*/
	inline const LogColor LightGreen()	{return LogColor(NONE,LIGHT,NONE);}
	inline const LogColor Green()		{return LogColor(NONE,MEDIUM,NONE);}
	inline const LogColor DarkGreen()	{return LogColor(NONE,DARK,NONE);}
	/*blue*/
	inline const LogColor LightBlue()	{return LogColor(NONE,NONE,LIGHT);}
	inline const LogColor Blue()		{return LogColor(NONE,NONE,MEDIUM);}
	inline const LogColor DarkBlue()		{return LogColor(NONE,NONE,DARK);}
	/*magenta*/
	inline const LogColor LightMagenta()	{return LogColor(LIGHT,NONE,LIGHT);}
	inline const LogColor Magenta()		{return LogColor(MEDIUM,NONE,MEDIUM);}
	inline const LogColor DarkMagenta()	{return LogColor(DARK,NONE,DARK);}
	/*cyan*/
	inline const LogColor LightCyan()	{return LogColor(NONE,LIGHT,LIGHT);}
	inline const LogColor Cyan()		{return LogColor(NONE,NONE,MEDIUM);}
	inline const LogColor DarkCyan()		{return LogColor(NONE,DARK,DARK);}
	/*yellow*/
	inline const LogColor LightYellow()	{return LogColor(LIGHT,LIGHT,NONE);}
	inline const LogColor Yellow()		{return LogColor(MEDIUM,MEDIUM,NONE);}
	inline const LogColor DarkYellow()	{return LogColor(DARK,DARK,NONE);}
	/*orange*/
	inline const LogColor Orange()		{return LogColor(LIGHT,MEDIUM,NONE);}
	inline const LogColor DarkOrange()	{return LogColor(MEDIUM,DARK,NONE);}
	/******/
}


