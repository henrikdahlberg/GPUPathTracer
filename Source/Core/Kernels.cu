#include <Core/Geometry.h>
#include <Core/Camera.h>
#include <Core/Scene.h>
#include <Core/Image.h>
#include <Shapes/Sphere.h>
#include <Utility/MathUtility.h>

#include <Core/Include.h>

//////////////////////////////////////////////////////////////////////////
// Settings, TODO: Move to proper places
//////////////////////////////////////////////////////////////////////////
#define BLOCK_SIZE 128
#define MAX_RAY_DEPTH 11 // Should probably be part of the HRenderer
#define STREAM_COMPACTION	

using namespace HMathUtility;

// Used to convert color to a format that OpenGL can display
// Represents the color in memory as either 1 float or 4 chars (32 bits)
union HColor {
	float value;
	struct { unsigned char x, y, z, w; } components;
};

// Stream compaction predicate
struct IsNegative {
	__host__ __device__ bool operator()(const int &x) { return x < 0; }
};

namespace HKernels {

	//////////////////////////////////////////////////////////////////////////
	// Device Kernels
	//////////////////////////////////////////////////////////////////////////

	__device__ glm::vec3 HemisphereCosSample(const glm::vec3 normal,
											 const float r1,
											 const float r2) {
		float c = sqrtf(r1);
		float phi = r2 * M_2PI;

		glm::vec3 w = fabs(normal.x) < M_SQRT1_3 ? glm::vec3(1, 0, 0) : (fabs(normal.y) < M_SQRT1_3 ? glm::vec3(0, 1, 0) : glm::vec3(0, 0, 1));
		glm::vec3 u = normalize(cross(normal, w));
		glm::vec3 v = cross(normal, u);

		return sqrtf(1.0f - r1) * normal + (cosf(phi) * c * u) + (sinf(phi) * c * v);

	}

	//////////////////////////////////////////////////////////////////////////
	// Global Kernels
	//////////////////////////////////////////////////////////////////////////

	__global__ void InitData(unsigned int numPixels,
							 int* livePixels,
							 glm::vec3* colorMask,
							 glm::vec3* accumulatedColor) {
		int i = blockDim.x * blockIdx.x + threadIdx.x;

		if (i < numPixels) {
			livePixels[i] = i;
			colorMask[i] = glm::vec3(1.0f, 1.0f, 1.0f);
			accumulatedColor[i] = glm::vec3(0.0f, 0.0f, 0.0f);
		}
	}

	__global__ void InitCameraRays(HRay* rays,
								   HCameraData* cameraData,
								   unsigned int currentSeed) {
		int i = blockDim.x * blockIdx.x + threadIdx.x;

		if (i < cameraData->resolution.x*cameraData->resolution.y) {

			int x = i % cameraData->resolution.x;
			int y = cameraData->resolution.y - (i - x) / cameraData->resolution.x - 1;

			// Store camera coordinate system
			glm::vec3 position = cameraData->position;
			glm::vec3 forward = cameraData->forward;
			glm::vec3 right = cameraData->right;
			glm::vec3 up = cameraData->up;

			// Initialize random number generator
			thrust::default_random_engine rng(currentSeed + TWHash(i));
			thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);

			// Generate random pixel offsets for anti-aliasing
			// Expected value is 0.5 i.e. middle of pixel
			float dx = uniform(rng) - 0.5f;
			float dy = uniform(rng) - 0.5f;

			// Compute point on image plane and account for focal distance
			glm::vec3 pointOnImagePlane = position + ((forward
				+ (2.0f * (dx + x) / (cameraData->resolution.x - 1.0f) - 1.0f) * right * tanf(cameraData->FOV.x * M_PI_2 * M_1_180)
				+ (2.0f * (dy + y) / (cameraData->resolution.y - 1.0f) - 1.0f) * up * tanf(cameraData->FOV.y * M_PI_2 * M_1_180)))
				* cameraData->focalDistance;

			float apertureRadius = cameraData->apertureRadius;
			if (apertureRadius > M_EPSILON) {
				// Sample a point on the aperture
				float angle = M_2PI * uniform(rng);
				float distance = apertureRadius * sqrtf(uniform(rng));

				position += (cosf(angle) * right + sinf(angle) * up) * distance;
			}

			rays[i].origin = position;
			rays[i].direction = normalize(pointOnImagePlane - position);
		}
	}

	__global__ void TraceKernel(glm::vec3* accumulatedColor,
								glm::vec3* colorMask,
								int numLivePixels,
								int* livePixels,
								HRay* rays,
								HSphere* spheres,
								int numSpheres,
								HTriangle* triangles,
								int numTriangles,
								int currentSeed) {

		int i = blockDim.x * blockIdx.x + threadIdx.x;

#if !(defined(_WIN64) && defined(STREAM_COMPACTION))
		if (livePixels[i] == -1) return;
#endif

		if (i < numLivePixels) {

			int pixelIdx = livePixels[i];

			// Initialize random number generator
			thrust::default_random_engine rng(TWHash(pixelIdx) * currentSeed);
			thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);

			// Sphere intersection
			float t = M_INF;
			HSurfaceInteraction intersection;
			int nearestSphereIdx;
			int nearestTriIdx;
			bool nearestIsTri = false;

			for (int sphereIdx = 0; sphereIdx < numSpheres; sphereIdx++) {
				// Check ray for sphere intersection
				if (spheres[sphereIdx].Intersect(rays[pixelIdx], t, intersection)) {
					nearestSphereIdx = sphereIdx;
				}
			}

			for (int triIdx = 0; triIdx < numTriangles; triIdx++) {
				if (triangles[triIdx].Intersect(rays[pixelIdx], t, intersection)) {
					nearestTriIdx = triIdx;
					nearestIsTri = true;
				}
			}

			if (t < M_INF) {

				HMaterial material;
				if (nearestIsTri) {
					material = triangles[nearestTriIdx].material;
				}
				else {
					material = spheres[nearestSphereIdx].material;
				}

				// diffuse, emission, TODO: Specular etc
				accumulatedColor[pixelIdx] += colorMask[pixelIdx] * material.emission;
				colorMask[pixelIdx] *= material.diffuse;

				// TODO: Fix normal directions if bouncing from other side
				// TODO: BSDF etc
				// TODO: Handle roundoff errors properly to avoid self-intersection instead of a fixed offset
				//		 See PBRT v3, new chapter draft @http://pbrt.org/fp-error-section.pdf

				// TEMP Backface checking and normal flipping:
				// Instead of if-statement, just multiply normal by sign of dot(...), might help thread divergence
				if (dot(-rays[pixelIdx].direction, intersection.normal) < 0.0f) {
					intersection.normal = -1.0f * intersection.normal;
				}

				// Compute new ray direction and origin
				rays[pixelIdx].origin = intersection.position + 0.005f * intersection.normal;
				rays[pixelIdx].direction = HemisphereCosSample(intersection.normal,
															   uniform(rng),
															   uniform(rng));

			}
			else {
				// Add background color
				accumulatedColor[pixelIdx] += colorMask[pixelIdx] * 0.0f * glm::vec3(0.69f, 0.86f, 0.89f);
				colorMask[pixelIdx] = glm::vec3(0.0f);
			}

			if (length(colorMask[pixelIdx]) < M_EPSILON) {
				// Mark ray for termination
				livePixels[i] = -1;
			}
		}
	}

	__global__ void AccumulateKernel(glm::vec3* pixels,
									 glm::vec3* accumulationBuffer,
									 glm::vec3* accumulatedColor,
									 HCameraData* cameraData,
									 unsigned int passCounter) {
		int i = blockDim.x * blockIdx.x + threadIdx.x;

		if (i < cameraData->resolution.x * cameraData->resolution.y) {

			int x = i % cameraData->resolution.x;
			int y = cameraData->resolution.y - (i - x) / cameraData->resolution.x - 1;

			accumulationBuffer[i] = (accumulationBuffer[i] * (float)(passCounter - 1) + accumulatedColor[i]) / (float)passCounter;

			HColor color;
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
							unsigned int height) {

		int i = blockDim.x * blockIdx.x + threadIdx.x;

		if (i < width*height) {
			HColor color;
			color.value = pixels[i].z;

			colorBytes[3 * i] = (unsigned char)color.components.x;
			colorBytes[3 * i + 1] = (unsigned char)color.components.y;
			colorBytes[3 * i + 2] = (unsigned char)color.components.z;
		}
	}

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
									   int numTriangles) {
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

		// Generate new seed each millisecond from system time and ray depth
		SYSTEMTIME time;
		GetSystemTime(&time);
		long time_ms = (time.wSecond * 1000) + time.wMilliseconds;
		int hashedPassCounter = TWHash(passCounter);
		int currentSeed = hashedPassCounter + TWHash(time_ms);

		// Generate initial rays from camera
		InitCameraRays<<<gridSize, blockSize>>>(rays,
												cameraData,
												currentSeed);

		// Trace surviving rays until none left or maximum depth reached
		unsigned int newGridSize;
		for (int rayDepth = 0; rayDepth < MAX_RAY_DEPTH; rayDepth++) {

			// Compute new grid size accounting for number of live pixels
			newGridSize = (numLivePixels + blockSize - 1) / blockSize;

			// Generate new seed each millisecond from system time and ray depth
			GetSystemTime(&time);
			long time_ms = (time.wSecond * 1000) + time.wMilliseconds;
			currentSeed = hashedPassCounter + TWHash(time_ms) + rayDepth;

			TraceKernel<<<newGridSize, blockSize>>>(accumulatedColor,
													colorMask,
													numLivePixels,
													livePixels,
													rays,
													spheres,
													numSpheres,
													triangles,
													numTriangles,
													currentSeed);

			// TODO: Multi kernel ray propagation idea (less divergence per kernel):
			//			for each depth
			//				- trace for intersection
			//				- scatter and compute attenuation
			//				- compact away dead rays
			//			end

			// Remove terminated rays with stream compaction
#if defined(_WIN64) && defined(STREAM_COMPACTION)
			thrust::device_ptr<int> devPtr(livePixels);
			thrust::device_ptr<int> endPtr = thrust::remove_if(devPtr, devPtr + numLivePixels, IsNegative());
			numLivePixels = endPtr.get() - livePixels;
#endif

			// Debug print
			// TODO: Remove
			if (passCounter == 1) {
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
										unsigned int height) {
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
