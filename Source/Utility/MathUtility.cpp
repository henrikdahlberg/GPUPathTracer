#include "MathUtility.h"

__host__ __device__ extern unsigned int HMathUtility::TWHash(unsigned int s)
{
	s = (s ^ 61) ^ (s >> 16);
	s = s + (s << 3);
	s = s ^ (s >> 4);
	s = s * 0x27d4eb2d;
	s = s ^ (s >> 15);
	return s;
}
