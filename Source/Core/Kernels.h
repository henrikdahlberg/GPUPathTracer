#ifndef KERNELS_H
#define KERNELS_H

#include <Core/Camera.h>
#include <Core/Scene.h>
#include <Core/BVH.h>

namespace HKernels {

	// The 'extern "C"' declaration is necessary in order to call
	// CUDA kernels defined in .cu-files from .cpp files

	/**
	 * Main renderer call, does ray propagation and scattering
	 */
	extern "C" void LaunchRenderKernel(HImage* image,
									   glm::vec3* accumulatedColor,
									   glm::vec3* colorMask,
									   HCameraData* cameraData,
									   unsigned int passCounter,
									   HRay* rays,
									   HSphere* spheres,
									   unsigned int numSpheres,
									   BVHNode* rootNode,
									   HTriangle* triangles,
									   unsigned int numTriangles);

	/**
	 * Save GL context image stored in pixels to PNG. Called in HImage::SavePNG().
	 * 
	 * @param colorBytes	Target storage pointer for data to be saved to PNG
	 * @param pixels		GL context image. pixels.z contains color information
	 * @param resolution	The resolution of the image
	 */
	extern "C" void LaunchSavePNGKernel(unsigned char* colorBytes,
										glm::vec3* pixels,
										unsigned int width,
										unsigned int height);

}

#endif //KERNELS_H
