#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cuda_runtime.h>
#include <math.h>

struct HRay
{
	float3 Origin;
	float3 Direction;
};

#endif // GEOMETRY_H
