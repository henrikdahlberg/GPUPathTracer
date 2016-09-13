#include <Core/Geometry.h>
#include <Core/Camera.h>
#include <Core/Scene.h>
#include <Core/Image.h>
#include <Core/Medium.h>
#include <Shapes/Sphere.h>
#include <Utility/MathUtility.h>

#include <Core/Include.h>

//////////////////////////////////////////////////////////////////////////
// Settings, TODO: Move to proper places
//////////////////////////////////////////////////////////////////////////
#define BLOCK_SIZE 128
#define MAX_RAY_DEPTH 41 // Should probably be part of the HRenderer
#define STREAM_COMPACTION	

using namespace HMathUtility;

// Used to convert color to a format that OpenGL can display
// Represents the color in memory as either 1 float or 4 chars (32 bits)
union HColor {
	float value;
	struct { unsigned char x, y, z, w; } components;
};

// Stores computed Fresnel reflection and transmission
struct HFresnel {
	float reflection;
	float transmission;
};

// Stream compaction predicate
struct IsNegative {
	__host__ __device__ bool operator()(const int &x) { return x < 0; }
};

namespace HKernels {

	//////////////////////////////////////////////////////////////////////////
	// Device Kernels
	//////////////////////////////////////////////////////////////////////////
	__device__ glm::vec3 HemisphereCosSample(const glm::vec3 &normal,
											 const float r1,
											 const float r2) {
		float c = sqrtf(r1);
		float phi = r2 * M_2PI;

		glm::vec3 w = fabs(normal.x) < M_SQRT1_3 ? glm::vec3(1, 0, 0) : (fabs(normal.y) < M_SQRT1_3 ? glm::vec3(0, 1, 0) : glm::vec3(0, 0, 1));
		glm::vec3 u = normalize(cross(normal, w));
		glm::vec3 v = cross(normal, u);

		return sqrtf(1.0f - r1) * normal + (cosf(phi) * c * u) + (sinf(phi) * c * v);

	}

	__device__ glm::vec3 ScatterSample(const float r1, const float r2) {
		float cosTheta = 2.0f*r1 - 1.0f;
		float sinTheta = sqrtf(1 - cosTheta*cosTheta);
		float phi = M_2PI * r2;

		return glm::vec3(cosTheta, cosf(phi)*sinTheta, sinf(phi)*sinTheta);
	}

	__device__ glm::vec3 Transmission(glm::vec3 absorptionMultiplier, float distance) {
		glm::vec3 res;
		res.x = powf(M_E, -absorptionMultiplier.x * distance);
		res.y = powf(M_E, -absorptionMultiplier.y * distance);
		res.z = powf(M_E, -absorptionMultiplier.z * distance);
		return res;
	}
	
	__device__ inline glm::vec3 ReflectionDir(const glm::vec3 &normal,
											  const glm::vec3 &incident) {
		return 2.0f * dot(normal, incident) * normal - incident;
	}

	__device__ glm::vec3 TransmissionDir(const glm::vec3 &normal,
										 const glm::vec3 &incident,
										 float eta1, float eta2) {
		float cosTheta1 = dot(normal, incident);
		float r = eta1 / eta2;

		float radicand = 1.0f - powf(r, 2.0f) * (1.0f - powf(cosTheta1, 2.0f));

		if (radicand < 0.0f) { // total internal reflection
			return glm::vec3(0.0f); //temp, dont know what to do here
		}

		float cosTheta2 = sqrtf(radicand);
		return r*(-1.0f*incident) + (r*cosTheta1 - cosTheta2)*normal;
	}

	__device__ HFresnel fresnelEquations(const glm::vec3 &normal,
										 const glm::vec3 &incidentDir,
										 float eta1, float eta2,
										 const glm::vec3 &reflectionDir,
										 const glm::vec3 &transmissionDir) {
		HFresnel fresnel;

		float cosTheta1 = dot(normal, incidentDir);
		float cosTheta2 = dot(-normal, transmissionDir);

		float s1 = eta1*cosTheta1;
		float s2 = eta2*cosTheta2;
		float p1 = eta1*cosTheta2;
		float p2 = eta2*cosTheta1;
		
		// Average s- and p-polarization
		fresnel.reflection = 0.5f*(powf((s1-s2)/(s1+s2), 2.0f) + powf((p1-p2)/(p1+p2), 2.0f));
		fresnel.transmission = 1.0f - fresnel.reflection;
		return fresnel;
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
			glm::vec3 pointOnImagePlane = position + (forward
				+ (2.0f * (dx + x) / (cameraData->resolution.x - 1.0f) - 1.0f) * right * tanf(cameraData->FOV.x * M_PI_2 * M_1_180)
				+ (2.0f * (dy + y) / (cameraData->resolution.y - 1.0f) - 1.0f) * up * tanf(cameraData->FOV.y * M_PI_2 * M_1_180))
				* cameraData->focalDistance;

			float apertureRadius = cameraData->apertureRadius;
			if (apertureRadius > M_EPSILON) {
				// Sample a point on the aperture
				float angle = M_2PI * uniform(rng);
				float distance = apertureRadius * sqrtf(uniform(rng));

				position += (cosf(angle) * right + sinf(angle) * up) * distance;
			}

			// Initialize ray
			HRay ray;
			ray.origin = position;
			ray.direction = normalize(pointOnImagePlane - position);
			ray.enteredMedium = HMedium();
			ray.currentMedium = HMedium();
			ray.transmitted = false;
			rays[i] = ray;
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

			HRay currentRay = rays[pixelIdx];

			// Initialize random number generator
			thrust::default_random_engine rng(TWHash(pixelIdx) * currentSeed);
			thrust::uniform_real_distribution<float> uniform(0.0f, 1.0f);

			// Intersection variables
			float t = M_INF;
			HSurfaceInteraction intersection;
			int nearestSphereIdx;
			int nearestTriIdx;
			bool nearestIsTri = false;

			// Sphere intersection
			for (int sphereIdx = 0; sphereIdx < numSpheres; sphereIdx++) {
				if (spheres[sphereIdx].Intersect(currentRay, t, intersection)) {
					nearestSphereIdx = sphereIdx;
				}
			}

			// Triangle intersection
			for (int triIdx = 0; triIdx < numTriangles; triIdx++) {
				if (triangles[triIdx].Intersect(currentRay, t, intersection)) {
					nearestTriIdx = triIdx;
					nearestIsTri = true;
				}
			}

			// Subsurface scattering.
			HScatteringProperties scattering = currentRay.currentMedium.scatteringProperties;
			if (scattering.reducedScatteringCoefficient > 0 ||
				dot(scattering.absorptionCoefficient, scattering.absorptionCoefficient) > M_EPSILON) {
				float scatteringDistance = -log(uniform(rng)) / scattering.reducedScatteringCoefficient;
				if (scatteringDistance < t) {
					// Scattering
					currentRay.origin = currentRay(scatteringDistance);
					currentRay.direction = ScatterSample(uniform(rng), uniform(rng));
					rays[pixelIdx] = currentRay;

					// Absorption
					colorMask[pixelIdx] *= Transmission(scattering.absorptionCoefficient, scatteringDistance);

					if (length(colorMask[pixelIdx]) < M_EPSILON) {
						// Mark ray for termination
						livePixels[i] = -1;
					}

					return;
				}
				else {
					// Absorption
					colorMask[pixelIdx] *= Transmission(scattering.absorptionCoefficient, t);
				}
			}

			// Treat intersection
			if (t < M_INF) {

				// Retreive intersection material
				HMaterial material;
				if (nearestIsTri) {
					material = triangles[nearestTriIdx].material;
				}
				else {
					material = spheres[nearestSphereIdx].material;
				}
				
				// TODO: Handle roundoff errors properly to avoid self-intersection instead of a fixed offset
				//		 See PBRT v3, new chapter draft @http://pbrt.org/fp-error-section.pdf
				glm::vec3 incidentDir = -currentRay.direction;

				// TEMP Backface checking and normal flipping:
				if (dot(incidentDir, intersection.normal) < 0.0f) {
					intersection.normal = -1.0f * intersection.normal;
				}

				// TODO: After an intersection is found, do the scattering in a separate kernel instead
				HMedium incidentMedium = currentRay.currentMedium;
				HMedium transmittedMedium;
				// Here we handle the assigning of medium based on if the ray has been transmitted or not
				// We assume that all transmissive materials are closed disjoint manifolds
				if (currentRay.transmitted) {
					// Ray is coming from inside of the object it has entered
					transmittedMedium = currentRay.enteredMedium;
				}
				else {
					// Ray is approaching an object
					transmittedMedium = material.medium;
				}

				// Compute reflection and transmission directions using Snell's law
				glm::vec3 reflectionDir = ReflectionDir(intersection.normal, incidentDir);
				glm::vec3 transmissionDir = TransmissionDir(intersection.normal, incidentDir,
															incidentMedium.eta,
															transmittedMedium.eta);

				// Russian roulette sampling of specular reflection
				bool doReflect = (material.materialType & SPECULAR) &&
					(uniform(rng) < fresnelEquations(intersection.normal,
													 incidentDir,
													 incidentMedium.eta,
													 transmittedMedium.eta,
													 reflectionDir,
													 transmissionDir).reflection);

				// Based on defined material properties, scatter the ray
				if (doReflect || material.materialType & REFLECTION) { // reflection
					colorMask[pixelIdx] *= material.specular;

					currentRay.origin = intersection.position + 0.001f * intersection.normal;
					currentRay.direction = reflectionDir;
					rays[pixelIdx] = currentRay;
				}
				else if (material.materialType & TRANSMISSION) { // transmission
					currentRay.origin = intersection.position - 0.001f * intersection.normal;
					currentRay.direction = transmissionDir;
					currentRay.enteredMedium = currentRay.currentMedium;
					currentRay.currentMedium = transmittedMedium;
					currentRay.transmitted = !currentRay.transmitted;
					rays[pixelIdx] = currentRay;
				}
				else { // diffuse
					accumulatedColor[pixelIdx] += colorMask[pixelIdx] * material.emission;
					colorMask[pixelIdx] *= material.diffuse;

					// Compute new ray direction and origin
					currentRay.origin = intersection.position + 0.001f * intersection.normal;
					currentRay.direction = HemisphereCosSample(intersection.normal,
															   uniform(rng),
															   uniform(rng));
					rays[pixelIdx] = currentRay;
				}
			}
			else {
				// The ray didn't intersect the scene, add background color and terminate ray
				accumulatedColor[pixelIdx] += colorMask[pixelIdx] * 0.5f * (1.0f - 0.5f * fabs(dot(currentRay.direction, glm::vec3(0.0f, 1.0f, 0.0f)))) * glm::vec3(0.69f, 0.86f, 0.89f);
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

			// Convert to 32-bit color
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

		// Initialize ray properties
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

			// Ray propagation kernel
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

			// Remove terminated rays with stream compaction
#if defined(_WIN64) && defined(STREAM_COMPACTION)
			thrust::device_ptr<int> devPtr(livePixels);
			thrust::device_ptr<int> endPtr = thrust::remove_if(devPtr, devPtr + numLivePixels, IsNegative());
			numLivePixels = endPtr.get() - livePixels;
#endif

#ifndef NDEBUG
			if (passCounter == 1) {
				std::cout << "Current Ray depth: " << rayDepth << std::endl;
				std::cout << "Number of live rays: " << numLivePixels << std::endl;
				std::cout << "Number of thread blocks: " << newGridSize << std::endl;
			}
#endif // NDEBUG
		}

		// TODO: Move the accumulation and OpenGL interoperability into the core loop somehow
		// Perform color conversion and gamma correction and pass computed colors to image
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
