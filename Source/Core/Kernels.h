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
	extern "C" void LaunchRenderKernel(
		HImage* Image,
		HCameraData* CameraData,
		unsigned int PassCounter,
		HRay* Rays,
		HSphere* Spheres,
		unsigned int NumSpheres);

	/**
	 * Save GL context image stored in Pixels to PNG. Called in HImage::SavePNG().
	 * 
	 * @param ColorBytes	Target storage pointer for data to be saved to PNG
	 * @param Pixels		GL context image. Pixels.z contains color information
	 * @param Resolution	The resolution of the image
	 */
	extern "C" void LaunchSavePNGKernel(
		unsigned char* ColorBytes,
		float3* Pixels,
		uint2 Resolution);

}

#endif //KERNELS_H
