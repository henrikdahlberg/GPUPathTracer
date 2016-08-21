#ifndef KERNELS_H
#define KERNELS_H

#include "Camera.h"
#include "Scene.h"

namespace HKernels
{

	/**
	* Integer hash function by Thomas Wang. Used to permute the render pass counter to be 
	* used as seed in the curand random number generator on the GPU.
	* 
	* @param s				Seed to be hashed.
	* @return				The hashed seed.
	*/
	__device__ unsigned int TWHash(unsigned int s);

	// The 'extern "C"' declaration is necessary in order to call
	// CUDA kernels defined in .cu-files from .cpp files
	extern "C" void LaunchRenderKernel(HImage* image,
									   float3* accumulatedColor,
									   float3* colorMask,
									   HCameraData* cameraData,
									   unsigned int passCounter,
									   HRay* rays,
									   HSphere* spheres,
									   unsigned int numSpheres);

	/**
	 * Save GL context image stored in pixels to PNG. Called in HImage::SavePNG().
	 * 
	 * @param colorBytes	Target storage pointer for data to be saved to PNG
	 * @param pixels		GL context image. pixels.z contains color information
	 * @param resolution	The resolution of the image
	 */
	extern "C" void LaunchSavePNGKernel(unsigned char* colorBytes,
										float3* pixels,
										uint2 resolution);

}

#endif //KERNELS_H
