#ifndef MATHUTILITY_H
#define MATHUTILITY_H

#include <cuda.h>
#include <math.h>
#include <float.h>

#include <Utility/External/cutil_math.h>

#define M_EPSILON  0.00001f
#define M_INF	   FLT_MAX
#define M_E        2.71828182845904523536f
#define M_LOG2E    1.44269504088896340736f
#define M_LOG10E   0.434294481903251827651f
#define M_LN2      0.693147180559945309417f
#define M_LN10     2.30258509299404568402f
#define M_PI       3.14159265358979323846f
#define M_2PI      6.28318530717958647692f
#define M_PI_2     1.57079632679489661923f
#define M_PI_4     0.785398163397448309616f
#define M_1_PI     0.318309886183790671538f
#define M_2_PI     0.636619772367581343076f
#define M_2_SQRTPI 1.12837916709551257390f
#define M_SQRT2    1.41421356237309504880f
#define M_SQRT1_2  0.707106781186547524401f
#define M_SQRT1_3  0.577350269189625764509f
#define M_1_180    0.005555555555555555556f

namespace HMathUtility
{

	/**
	 * Convert angle in radians to angle in degrees.
	 * @param Rad	Angle in radians.
	 */
	extern inline float RadToDeg(float radians) { return radians * 180.0f * M_1_PI; }

	/**
	* Convert angle in degrees to angle in radians.
	* @param Deg	Angle in degrees.
	*/
	extern inline float DegToRad(float degrees) { return degrees * M_PI * M_1_180; }

}

#endif // MATHUTILITY_H
