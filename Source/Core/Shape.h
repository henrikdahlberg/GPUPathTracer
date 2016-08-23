#ifndef SHAPE_H
#define SHAPE_H

#include <cuda.h>
#include <cuda_runtime.h>

#include <Core/Geometry.h>
#include <Core/Material.h>
#include <Core/Interaction.h>

struct HShape
{

	__host__ __device__ bool Intersect(HRay &ray, float &t,
									   HSurfaceInteraction &intersection);

	HMaterial material;

};

#endif // SHAPE_H