#pragma once 


#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


///	\namespace SignalUtils
///	\brief some signal utilities gathered in a class
///	\author 
///	\note
namespace SignalUtils
{

	// set output to zero when fabs(input)< threshold
	inline double deadZone(double in, const double threshold);
	
	// set output to sign(input)*minValue when fabs(input)>=thresholdm fabs(input)<thresholdM
	inline double minValueZone(double in, const double minValue, const double thresholdm, const double thresholdM);


	template<typename T>
	inline T sat(T val, T min, T max) 
	{
		return std::min(std::max(val, min), max);
	}

	template<typename T>
	inline int sign(const T a)   
	{	
		return ( (a >= 0) ? 1 : -1  );
	}

};



// set output to zero when fabs(input)< threshold
double SignalUtils::deadZone(double in, double threshold)
{
	double out = in;
	if(fabs(in) < threshold)
	{
		out = 0;
	}
	return out;
}

// set output to sign(input)*value when fabs(input)>=thresholdm fabs(input)<thresholdM
double	SignalUtils::minValueZone(double in, const double minValue, const double thresholdm, const double thresholdM)
{
	double out 	= in;
	double fabsIn 	= fabs(in);

	if( (fabsIn < thresholdM) && (fabsIn >= thresholdm) )
	{
		out = sign(in) * fabs(minValue);
	}
	else if (fabsIn < thresholdm)
		out = 0;

	return out;
	
}
