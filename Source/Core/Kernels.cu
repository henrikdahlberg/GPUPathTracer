#include <Core/Geometry.h>
#include <Core/Camera.h>
#include <Core/Scene.h>
#include <Core/Image.h>
#include <Shapes/Sphere.h>
#include <Utility/MathUtility.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <math.h>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif // GLM_FORCE_CUDA
#include <glm/glm.hpp>

//////////////////////////////////////////////////////////////////////////
// Settings, TODO: Move to proper places
//////////////////////////////////////////////////////////////////////////
#define BLOCK_SIZE 128
#define MAX_RAY_DEPTH 11 // Should probably be part of the HRenderer
#define STREAM_COMPACTION	

// Used to convert color to a format that OpenGL can display
// Represents the color in memory as either 1 float or 4 chars (32 bits)
union HColor
{
	float value;
	struct
	{
		unsigned char x, y, z, w;
	} components;
	//uchar4 components;
};

namespace HKernels
{

	//////////////////////////////////////////////////////////////////////////
	// Device Kernels
	//////////////////////////////////////////////////////////////////////////
	__device__ unsigned int TWHash(unsigned int s)
	{
		s = (s ^ 61) ^ (s >> 16);
		s = s + (s << 3);
		s = s ^ (s >> 4);
		s = s * 0x27d4eb2d;
		s = s ^ (s >> 15);
		return s;
	}

	__device__ glm::vec3 HemisphereCosSample(const glm::vec3 normal,
										  const float r1,
										  const float r2)
	{

		float c = sqrtf(r1);
		float s = sqrtf(1.0f - c*c);
		float t = r2 * M_2PI;

		glm::vec3 w = fabs(normal.x) < M_SQRT1_3 ? glm::vec3(1, 0, 0) : (fabs(normal.y) < M_SQRT1_3 ? glm::vec3(0, 1, 0) : glm::vec3(0, 0, 1));
		glm::vec3 u = normalize(cross(normal, w));
		glm::vec3 v = cross(normal, u);

		return c * normal + (__cosf(t) * s * u) + (__sinf(t) * s * v);

	}

	//////////////////////////////////////////////////////////////////////////
	// Global Kernels
	//////////////////////////////////////////////////////////////////////////

	__global__ void InitData(unsigned int numPixels,
							 int* livePixels,
							 glm::vec3* colorMask,
							 glm::vec3* accumulatedColor)
	{

		int i = blockDim.x * blockIdx.x + threadIdx.x;

		if (i < numPixels)
		{

			livePixels[i] = i;
			colorMask[i] = glm::vec3(1.0f, 1.0f, 1.0f);
			accumulatedColor[i] = glm::vec3(0.0f, 0.0f, 0.0f);

		}

	}

	__global__ void InitCameraRays(HRay* rays,
								   HCameraData* cameraData,
								   unsigned int passCounter)
	{

		int i = blockDim.x * blockIdx.x + threadIdx.x;

		if (i < cameraData->resolution.x*cameraData->resolution.y)
		{

			int x = i % cameraData->resolution.x;
			int y = cameraData->resolution.y - (i - x) / cameraData->resolution.x - 1;

			// TODO: Maybe the camera axis computations should be handled CPU-side
			// stored and updated only when the camera is moved
			glm::vec3 position = cameraData->position;
			glm::vec3 view = normalize(cameraData->view); // Shouldn't need normalization

			// Compute horizontal and vertical axes on camera image plane
			glm::vec3 horizontalAxis = normalize(cross(view, cameraData->up));
			glm::vec3 verticalAxis = normalize(cross(horizontalAxis, view));

			// Compute middle point on camera image plane
			glm::vec3 middle = position + view;

			// Initialize random number generator
			curandState RNGState;
			curand_init(TWHash(passCounter) + TWHash(i), 0, 0, &RNGState);

			// Generate random pixel offsets for anti-aliasing
			// Expected value is 0.5 i.e. middle of pixel
			float OffsetX = curand_uniform(&RNGState) - 0.5f;
			float OffsetY = curand_uniform(&RNGState) - 0.5f;

			// Compute point on image plane and account for focal distance
			glm::vec3 pointOnImagePlane = position + ((middle
				+ (2.0f * (OffsetX + x) / (cameraData->resolution.x - 1.0f) - 1.0f) * horizontalAxis * __tanf(cameraData->FOV.x * M_PI_2 * M_1_180)
				+ (2.0f * (OffsetY + y) / (cameraData->resolution.y - 1.0f) - 1.0f) * verticalAxis * __tanf(cameraData->FOV.y * M_PI_2 * M_1_180)) - position)
				* cameraData->focalDistance;

			float apertureRadius = cameraData->apertureRadius;
			if (apertureRadius > M_EPSILON)
			{
				// Sample a point on the aperture
				float angle = M_2PI * curand_uniform(&RNGState);
				float distance = apertureRadius * sqrtf(curand_uniform(&RNGState));

				position += (__cosf(angle) * horizontalAxis + __sinf(angle) * verticalAxis) * distance;
			}

			rays[i].origin = position;
			rays[i].direction = normalize(pointOnImagePlane - position);

		}

	}

__global__ void TraceKernel(glm::vec3* accumulatedColor,
							glm::vec3* colorMask,
							int numLivePixels,
							int* livePixels,
							unsigned int passCounter,
							HRay* rays,
							HSphere* spheres,
							int numSpheres,
							HTriangle* triangles,
							int numTriangles)
{

	int i = blockDim.x * blockIdx.x + threadIdx.x;

#if !(defined(_WIN64) && defined(STREAM_COMPACTION))
	if (livePixels[i] == -1) return;
#endif

	if (i < numLivePixels)
	{

		int pixelIdx = livePixels[i];

		// Initialize random number generator
		curandState RNGState;
		curand_init(TWHash(passCounter) + i, 0, 0, &RNGState);

		// Sphere intersection
		float t = M_INF;
		HSurfaceInteraction intersection;
		int nearestSphereIdx;
		int nearestTriIdx;
		bool nearestIsTri = false;

		for (int sphereIdx = 0; sphereIdx < numSpheres; sphereIdx++)
		{

			// Check ray for sphere intersection
			if (spheres[sphereIdx].Intersect(rays[pixelIdx], t, intersection))
			{
				nearestSphereIdx = sphereIdx;
			}

		}
		
		for (int triIdx = 0; triIdx < numTriangles; triIdx++)
		{
			if (triangles[triIdx].Intersect(rays[pixelIdx], t, intersection))
			{
				nearestTriIdx = triIdx;
				nearestIsTri = true;
			}
		}

		if (t < M_INF)
		{

			HMaterial material;
			if (nearestIsTri)
			{
				 material = triangles[nearestTriIdx].material;
			}
			else
			{
				material = spheres[nearestSphereIdx].material;
			}

			// diffuse, emission, TODO: Specular etc
			accumulatedColor[pixelIdx] += colorMask[pixelIdx] * material.emission;
			colorMask[pixelIdx] *= material.diffuse;

			// Compute new ray direction
			// TODO: BSDF etc
			// TODO: Handle roundoff errors properly to avoid self-intersection instead of a fixed offset
			//		 See PBRT v3, new chapter draft @http://pbrt.org/fp-error-section.pdf
			rays[pixelIdx].origin = intersection.position + 0.005f * intersection.normal;
			rays[pixelIdx].direction = HemisphereCosSample(intersection.normal,
														   curand_uniform(&RNGState),
														   curand_uniform(&RNGState));

		}
		else
		{

			// Add background color
			accumulatedColor[pixelIdx] += colorMask[pixelIdx] * glm::vec3(0.3f);
			colorMask[pixelIdx] = glm::vec3(0.0f);

		}

		if (length(colorMask[pixelIdx]) < M_EPSILON)
		{

			// Mark ray for termination
			livePixels[i] = -1;

		}

	}

}

	__global__ void AccumulateKernel(glm::vec3* pixels,
									 glm::vec3* accumulationBuffer,
									 glm::vec3* accumulatedColor,
									 HCameraData* cameraData,
									 unsigned int passCounter)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;

		if (i < cameraData->resolution.x * cameraData->resolution.y)
		{

			int x = i % cameraData->resolution.x;
			int y = cameraData->resolution.y - (i - x) / cameraData->resolution.x - 1;

			accumulationBuffer[i] = (accumulationBuffer[i] * (float)(passCounter - 1) + accumulatedColor[i]) / (float)passCounter;

			HColor color;
			//color.components = make_uchar4((unsigned char)(powf(clamp(accumulationBuffer[i].x, 0.0f, 1.0f), 1 / 2.2f) * 255),
			//							   (unsigned char)(powf(clamp(accumulationBuffer[i].y, 0.0f, 1.0f), 1 / 2.2f) * 255),
			//							   (unsigned char)(powf(clamp(accumulationBuffer[i].z, 0.0f, 1.0f), 1 / 2.2f) * 255),
			//							   1);
			color.components.x = (unsigned char)(powf(clamp(accumulationBuffer[i].x, 0.0f, 1.0f), 1 / 2.2f) * 255);
			color.components.y = (unsigned char)(powf(clamp(accumulationBuffer[i].y, 0.0f, 1.0f), 1 / 2.2f) * 255);
			color.components.z = (unsigned char)(powf(clamp(accumulationBuffer[i].z, 0.0f, 1.0f), 1 / 2.2f) * 255);
			color.components.w = 1;

			// Pass pixel coordinates and pixel color in OpenGL to output buffer
			pixels[i] = glm::vec3(x, y, color.value);

		}

	}

	__global__ void SavePNG(unsigned char* colorBytes,
							glm::vec3* pixels,
							unsigned int width,
							unsigned int height)
	{

		int i = blockDim.x * blockIdx.x + threadIdx.x;

		if (i < width*height)
		{

			HColor color;
			color.value = pixels[i].z;

			colorBytes[3 * i    ] = (unsigned char)color.components.x;
			colorBytes[3 * i + 1] = (unsigned char)color.components.y;
			colorBytes[3 * i + 2] = (unsigned char)color.components.z;

		}

	}

	// Stream compaction predicate
	struct IsNegative
	{
		__host__ __device__ bool operator()(const int &x)
		{
			return x < 0;
		}
	};

	//////////////////////////////////////////////////////////////////////////
	// External CUDA access launch function
	//////////////////////////////////////////////////////////////////////////
	extern "C" void LaunchRenderKernel(HImage* image,
									   glm::vec3* accumulatedColor,
									   glm::vec3* colorMask,
									   HCameraData* cameraData,
									   unsigned int passCounter,
									   HRay* rays,
									   HSphere* spheres,
									   unsigned int numSpheres,
									   HTriangle* triangles,
									   int numTriangles)
	{

		unsigned int blockSize = BLOCK_SIZE;
		unsigned int gridSize = (image->numPixels + blockSize - 1) / blockSize;

		unsigned int numLivePixels = image->numPixels;
		int* livePixels = nullptr;

		// Inefficient to do this every call but fine until I figure out
		// how to resize allocated memory on device (after stream compaction)
		checkCudaErrors(cudaMalloc(&livePixels, image->numPixels*sizeof(int)));

		// TODO: Combine these initialization kernels to avoid one kernel launch
		InitData<<<gridSize, blockSize>>>(numLivePixels,
										  livePixels,
										  colorMask,
										  accumulatedColor);

		// Generate initial rays from camera
		InitCameraRays<<<gridSize, blockSize>>>(rays,
												cameraData,
												passCounter);

		// Trace surviving rays until none left or maximum depth reached
		unsigned int newGridSize;
		for (int rayDepth = 0; rayDepth < MAX_RAY_DEPTH; rayDepth++)
		{
			
			// Compute new grid size accounting for number of live pixels
			newGridSize = (numLivePixels + blockSize - 1) / blockSize;

			TraceKernel<<<newGridSize, blockSize>>>(accumulatedColor,
													colorMask,
													numLivePixels,
													livePixels,
													passCounter,
													rays,
													spheres,
													numSpheres,
													triangles,
													numTriangles);

			// Remove terminated rays with stream compaction
#if defined(_WIN64) && defined(STREAM_COMPACTION)
			thrust::device_ptr<int> devPtr(livePixels);
			thrust::device_ptr<int> endPtr = thrust::remove_if(devPtr, devPtr + numLivePixels, IsNegative());
			numLivePixels = endPtr.get() - livePixels;
#endif

			// Debug print
			// TODO: Remove
			if (passCounter == 1)
			{
				std::cout << "Current Ray depth: " << rayDepth << std::endl;
				std::cout << "Number of live rays: " << numLivePixels << std::endl;
				std::cout << "Number of thread blocks: " << newGridSize << std::endl;
			}

		}

		// TODO: Move the accumulation and OpenGL interoperability into the core loop somehow
		AccumulateKernel<<<gridSize, blockSize>>>(image->pixels,
												  image->accumulationBuffer,
												  accumulatedColor,
												  cameraData,
												  passCounter);

		checkCudaErrors(cudaFree(livePixels));

	}

	extern "C" void LaunchSavePNGKernel(unsigned char* colorBytes,
										glm::vec3* pixels,
										unsigned int width,
										unsigned int height)
	{

		unsigned int blockSize = BLOCK_SIZE;
 		unsigned int gridSize = (width*height + blockSize - 1) / blockSize;

		checkCudaErrors(cudaDeviceSynchronize());
		SavePNG<<<gridSize, blockSize>>>(colorBytes,
										 pixels,
										 width,
										 height);
		checkCudaErrors(cudaDeviceSynchronize());

	}

}
