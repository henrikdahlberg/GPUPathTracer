#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cuda_runtime.h>
#include <math.h>

#include "Material.h"

struct HRay
{

	float3 Origin;
	float3 Direction;

};

struct HSphere
{

	float3 Position;
	float Radius;

	HMaterial* Material;

};

#endif // GEOMETRY_H
