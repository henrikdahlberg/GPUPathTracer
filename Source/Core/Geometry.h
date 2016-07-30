#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cuda_runtime.h>
#include <math.h>

struct HRay
{
	float3 Origin;
	float3 Direction;
};

struct HSphere
{
	float3 Origin;
	float Radius;
};

#endif // GEOMETRY_H
