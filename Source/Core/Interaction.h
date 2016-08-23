#ifndef INTERACTION_H
#define INTERACTION_H

#include <cuda.h>
#include <cuda_runtime.h>

#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif // GLM_FORCE_CUDA
#include <glm/glm.hpp>

struct HInteraction
{

	__host__ __device__ HInteraction()
	{

	}

	__host__ __device__ HInteraction(const glm::vec3 &p)
		: position(p)
	{

	}

	glm::vec3 position;

};

struct HSurfaceInteraction : HInteraction
{
	
	__host__ __device__ HSurfaceInteraction()
	{

	}

	__host__ __device__ HSurfaceInteraction(const glm::vec3 &p, const glm::vec3 &n)
		: HInteraction(p), normal(n)
	{

	}

	glm::vec3 normal;

};

struct HVolumeInteraction : HInteraction
{
	// TODO: Subsurface scattering, volumetric scattering
};

#endif // INTERACTION_H