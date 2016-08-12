#include "Geometry.h"
#include "Camera.h"
#include "Scene.h"
#include "Image.h"
#include "Utility/MathUtility.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
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

//////////////////////////////////////////////////////////////////////////
// Settings, TODO: Move to proper places
//////////////////////////////////////////////////////////////////////////
#define BLOCK_SIZE 256
#define MAX_RAY_DEPTH 11 // Should probably be part of the HRenderer
//#define STREAM_COMPACTION

// Used to convert color to a format that OpenGL can display
// Represents the color in memory as either 1 float or 4 chars (32 bits)
union HColor
{
	float Value;
	uchar4 Components;
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

	__device__ float IntersectSphere(
		const HSphere Sphere,
		const HRay Ray,
		float3 &IntersectionPoint,
		float3 &IntersectionNormal)
	{

		float3 OP = Sphere.Position - Ray.Origin;
		float t;
		float b = dot(OP, Ray.Direction);
		float Discriminant = b*b - dot(OP, OP) + Sphere.Radius*Sphere.Radius;

		if (Discriminant < 0)
		{
			return -1.0f;
		}
		
		Discriminant = sqrtf(Discriminant);

		float t1 = b - Discriminant;
		float t2 = b + Discriminant;

		if (t1 > M_EPSILON)
		{
			t = t1;
		}
		else if (t2 > M_EPSILON)
		{
			t = t2;
		}
		else
		{
			return -1.0f;
		}

		IntersectionPoint = Ray.Origin + t*Ray.Direction;
		IntersectionNormal = normalize(IntersectionPoint - Sphere.Position);

		return t;

	}

	__device__ float3 HemisphereCosSample(
		const float3 Normal,
		const float r1,
		const float r2)
	{

		float c = sqrtf(r1);
		float s = sqrtf(1.0f - c*c);
		float t = r2 * M_2PI;

		float3 w = fabs(Normal.x) < M_SQRT1_3 ? make_float3(1, 0, 0) : (fabs(Normal.y) < M_SQRT1_3 ? make_float3(0, 1, 0) : make_float3(0, 0, 1));
		float3 u = normalize(cross(Normal, w));
		float3 v = cross(Normal, u);

		return c * Normal + (cos(t) * s * u) + (sin(t) * s * v);

	}

	//////////////////////////////////////////////////////////////////////////
	// Global Kernels
	//////////////////////////////////////////////////////////////////////////

	// TODO: Delete, used for testing CUDA access to structs with pointers. Not fully working
	/*__global__ void TestSceneDataKernel(
		HSceneData* SceneData,
		unsigned int PassCounter)
	{
		int x = blockIdx.x*blockDim.x + threadIdx.x;
		int y = blockIdx.y*blockDim.y + threadIdx.y;

		if (x == 0 && PassCounter == 1)
		{
			//printf("%f \n", SceneData->Spheres[0].Radius);
			//printf("%d \n", SceneData->NumSpheres);
		}
	}*/

	__global__ void InitData(
		unsigned int NumPixels,
		int* LivePixels,
		float3* ColorMask,
		float3* AccumulatedColor)
	{

		int i = blockDim.x * blockIdx.x + threadIdx.x;

		if (i < NumPixels)
		{

			LivePixels[i] = i;
			ColorMask[i] = make_float3(1.0f, 1.0f, 1.0f);
			AccumulatedColor[i] = make_float3(0.0f, 0.0f, 0.0f);

		}

	}

	__global__ void InitCameraRays(
		HRay* Rays,
		HCameraData* CameraData,
		unsigned int PassCounter)
	{

		int i = blockDim.x * blockIdx.x + threadIdx.x;

		if (i < CameraData->Resolution.x*CameraData->Resolution.y)
		{

			int x = i % CameraData->Resolution.x;
			int y = CameraData->Resolution.y - (i - x) / CameraData->Resolution.x - 1;

			// TODO: Maybe the camera axis computations should be handled CPU-side
			// stored and updated only when the camera is moved
			float3 Position = CameraData->Position;
			float3 View = normalize(CameraData->View); // Shouldn't need normalization

			// Compute horizontal and vertical axes on camera image plane
			float3 HorizontalAxis = normalize(cross(View, CameraData->Up));
			float3 VerticalAxis = normalize(cross(HorizontalAxis, View));

			// Compute middle point on camera image plane
			float3 MiddlePoint = Position + View;
			float3 Horizontal = HorizontalAxis * tan(CameraData->FOV.x * M_PI_2 * M_1_180);
			float3 Vertical = VerticalAxis * tan(CameraData->FOV.y * M_PI_2 * M_1_180);

			// Initialize random number generator
			curandState RNGState;
			curand_init(TWHash(PassCounter) + i, 0, 0, &RNGState);

			// Generate random pixel offsets for anti-aliasing
			// Expected value is 0.5 i.e. middle of pixel
			float OffsetX = curand_uniform(&RNGState) - 0.5f;
			float OffsetY = curand_uniform(&RNGState) - 0.5f;

			// Compute point on image plane and account for focal distance
			float3 PointOnImagePlane = Position + ((MiddlePoint
				+ (2.0f * (OffsetX + x) / (CameraData->Resolution.x - 1.0f) - 1.0f) * Horizontal
				+ (2.0f * (OffsetY + y) / (CameraData->Resolution.y - 1.0f) - 1.0f) * Vertical) - Position)
				* CameraData->FocalDistance;

			float ApertureRadius = CameraData->ApertureRadius;
			if (ApertureRadius > M_EPSILON)
			{
				// Sample a point on the aperture
				float Angle = M_2PI * curand_uniform(&RNGState);
				float Distance = ApertureRadius * sqrtf(curand_uniform(&RNGState));

				Position += (cos(Angle) * HorizontalAxis + sin(Angle) * VerticalAxis) * Distance;
			}

			Rays[i].Origin = Position;
			Rays[i].Direction = normalize(PointOnImagePlane - Position);

		}

	}

	__global__ void TestRenderKernel(
		float3* Pixels,
		float3* AccumulationBuffer,
		HCameraData* CameraData,
		unsigned int PassCounter,
		HRay* Rays,
		HSphere* Spheres,
		unsigned int NumSpheres)
	{

		int i = blockDim.x * blockIdx.x + threadIdx.x;

		if (i < CameraData->Resolution.x*CameraData->Resolution.y)
		{

			int x = i % CameraData->Resolution.x;
			int y = CameraData->Resolution.y - (i - x) / CameraData->Resolution.x - 1;

			// Initialize random number generator
			curandState RNGState;
			curand_init(TWHash(PassCounter) + i, 0, 0, &RNGState);

			// Random test color
			//float3 TempColor = make_float3(curand_uniform(&RNGState), curand_uniform(&RNGState), curand_uniform(&RNGState));

			// Test Ray direction
			//float3 TempColor = make_float3(Rays[i].Direction.x, Rays[i].Direction.y, Rays[i].Direction.z);
			float3 TempColor = 0.5f + 0.5f*make_float3(Rays[i].Direction.x, Rays[i].Direction.y, -Rays[i].Direction.z);
			//float3 TempColor = curand_uniform(&RNGState) + TempColor*make_float3(Rays[i].Direction.x, Rays[i].Direction.y, -Rays[i].Direction.z);

			// Accumulate and average the color for each pass
			AccumulationBuffer[i] = (AccumulationBuffer[i] * (PassCounter - 1) + TempColor) / PassCounter;

			TempColor = AccumulationBuffer[i];

			// Make type conversion for OpenGL and perform gamma correction
			// TODO: Use sRGB instead of flat 2.2 gamma correction?
			HColor Color;
			Color.Components = make_uchar4(
				(unsigned char)(powf(TempColor.x, 1 / 2.2f) * 255),
				(unsigned char)(powf(TempColor.y, 1 / 2.2f) * 255),
				(unsigned char)(powf(TempColor.z, 1 / 2.2f) * 255), 1);

			// Pass pixel coordinates and pixel color in OpenGL to output buffer
			Pixels[i] = make_float3(x, y, Color.Value);

		}
		
	}

	__global__ void TraceKernel(
		float3* AccumulatedColor,
		float3* ColorMask,
		int NumLivePixels,
		int* LivePixels,
		unsigned int PassCounter,
		unsigned int RayDepth,
		HRay* Rays,		
		HSphere* Spheres,
		int NumSpheres)
	{

		int i = blockDim.x * blockIdx.x + threadIdx.x;

#if !(defined(_WIN64) && defined(STREAM_COMPACTION))
		if (LivePixels[i] == -1) return;
#endif

		if (i < NumLivePixels)
		{

			int PixelIdx = LivePixels[i];

			// Initialize random number generator
			curandState RNGState;
			curand_init(TWHash(PassCounter) + TWHash(i) + TWHash(RayDepth), 0, 0, &RNGState);

			float t;
			float3 IntersectionPoint;
			float3 IntersectionNormal;

			// Sphere intersection
			// TODO: More elegant way of saving intersections
			float tNearest = FLT_MAX;
			float3 NearestIntersectionPoint;
			float3 NearestIntersectionNormal;
			int NearestSphereIdx;

			for (int SphereIdx = 0; SphereIdx < NumSpheres; SphereIdx++)
			{

				// Check ray for sphere intersection,
				// store intersection point, normal and distance
				t = IntersectSphere(
					Spheres[SphereIdx],
					Rays[PixelIdx],
					IntersectionPoint,
					IntersectionNormal);

				if (t > 0 && t < tNearest)
				{

					// Make sure only the nearest intersection is kept
					tNearest = t;
					NearestIntersectionPoint = IntersectionPoint;
					NearestIntersectionNormal = IntersectionNormal;
					NearestSphereIdx = SphereIdx;

				}

			}

			if (tNearest < FLT_MAX)
			{

				HMaterial Material = Spheres[NearestSphereIdx].Material;

				// Diffuse, Emission, TODO: Specular etc
				AccumulatedColor[PixelIdx] += ColorMask[PixelIdx] * Material.Emission;
				ColorMask[PixelIdx] *= Material.Diffuse;

				// Compute new ray direction
				// TODO: BSDF etc
				// TODO: Handle roundoff errors properly to avoid self-intersection instead of a fixed offset
				//		 See PBRT v3, new chapter draft @http://pbrt.org/fp-error-section.pdf
				Rays[PixelIdx].Origin = NearestIntersectionPoint + 0.005f * NearestIntersectionNormal;
				Rays[PixelIdx].Direction = HemisphereCosSample(
					NearestIntersectionNormal,
					curand_uniform(&RNGState),
					curand_uniform(&RNGState));


			}
			else
			{

				// Add background color
				AccumulatedColor[PixelIdx] += ColorMask[PixelIdx] * make_float3(0.3f);
				ColorMask[PixelIdx] = make_float3(0.0f);

			}


			if (length(ColorMask[PixelIdx]) < M_EPSILON)
			{

				// Mark ray for termination
				LivePixels[i] = -1;

			}

		}

	}

	__global__ void AccumulateKernel(
		float3* Pixels,
		float3* AccumulationBuffer,
		float3* AccumulatedColors,
		HCameraData* CameraData,
		unsigned int PassCounter)
	{
		int i = blockDim.x * blockIdx.x + threadIdx.x;

		if (i < CameraData->Resolution.x * CameraData->Resolution.y)
		{

			int x = i % CameraData->Resolution.x;
			int y = CameraData->Resolution.y - (i - x) / CameraData->Resolution.x - 1;

			AccumulationBuffer[i] = (AccumulationBuffer[i] * (PassCounter - 1) + AccumulatedColors[i]) / PassCounter;

			HColor Color;
			Color.Components = make_uchar4(
				(unsigned char)(powf(clamp(AccumulationBuffer[i].x, 0.0f, 1.0f), 1 / 2.2f) * 255),
				(unsigned char)(powf(clamp(AccumulationBuffer[i].y, 0.0f, 1.0f), 1 / 2.2f) * 255),
				(unsigned char)(powf(clamp(AccumulationBuffer[i].z, 0.0f, 1.0f), 1 / 2.2f) * 255), 1);

			// Pass pixel coordinates and pixel color in OpenGL to output buffer
			Pixels[i] = make_float3(x, y, Color.Value);

		}

	}


	__global__ void SavePNG(
		unsigned char* ColorBytes,
		float3* Pixels,
		uint2 Resolution)
	{

		int i = blockDim.x * blockIdx.x + threadIdx.x;

		if (i < Resolution.x*Resolution.y)
		{

			HColor Color;
			Color.Value = Pixels[i].z;

			ColorBytes[3 * i    ] = (unsigned char)Color.Components.x;
			ColorBytes[3 * i + 1] = (unsigned char)Color.Components.y;
			ColorBytes[3 * i + 2] = (unsigned char)Color.Components.z;

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
	extern "C" void LaunchRenderKernel(
		HImage* Image,
		float3* AccumulatedColor,
		float3* ColorMask,
		HCameraData* CameraData,
		unsigned int PassCounter,
		HRay* Rays,
		HSphere* Spheres,
		unsigned int NumSpheres)
	{

		unsigned int BlockSize = BLOCK_SIZE;
		unsigned int GridSize = (Image->NumPixels + BlockSize - 1) / BlockSize;

		unsigned int NumLivePixels = Image->NumPixels;
		int* LivePixels = nullptr;

		// Inefficient to do this every call but fine until I figure out
		// how to resize allocated memory on device (after stream compaction)
		checkCudaErrors(cudaMalloc(&LivePixels, Image->NumPixels*sizeof(int)));

		// TODO: Combine these initialization kernels to avoid one kernel launch
		InitData<<<GridSize, BlockSize>>>(
			NumLivePixels,
			LivePixels,
			ColorMask,
			AccumulatedColor);

		// Generate initial rays from camera
		InitCameraRays<<<GridSize, BlockSize>>>(
			Rays,
			CameraData,
			PassCounter);

		// Temp testing
		/*TestRenderKernel<<<GridSize, BlockSize>>>(
			Image->Pixels,
			Image->AccumulationBuffer,
			CameraData,
			PassCounter,
			Rays,
			Spheres,
			NumSpheres);*/

		// Trace surviving rays until none left or maximum depth reached
		unsigned int NewGridSize;
		for (int RayDepth = 0; RayDepth < MAX_RAY_DEPTH; RayDepth++)
		{
			
			// Compute new grid size accounting for number of live pixels
			NewGridSize = (NumLivePixels + BlockSize - 1) / BlockSize;

			TraceKernel<<<NewGridSize, BlockSize>>>(
				AccumulatedColor,
				ColorMask,
				NumLivePixels,
				LivePixels,
				PassCounter,
				RayDepth,
				Rays,
				Spheres,
				NumSpheres);

			// Remove terminated rays with stream compaction
			// Only works in 64-bit build!
#if defined(_WIN64) && defined(STREAM_COMPACTION)
			thrust::device_ptr<int> DevPtr(LivePixels);
			thrust::device_ptr<int> EndPtr = thrust::remove_if(DevPtr, DevPtr + NumLivePixels, IsNegative());
			NumLivePixels = EndPtr.get() - LivePixels;
#endif

			// Debug print
			// TODO: Remove
			if (PassCounter == 1)
			{
				std::cout << "Current Ray depth: " << RayDepth << std::endl;
				std::cout << "Number of live rays: " << NumLivePixels << std::endl;
				std::cout << "Number of thread blocks: " << NewGridSize << std::endl;
			}

		}

		// TODO: Move the accumulation and OpenGL interoperability into the core loop somehow
		AccumulateKernel<<<GridSize, BlockSize>>>(
			Image->Pixels,
			Image->AccumulationBuffer,
			AccumulatedColor,
			CameraData,
			PassCounter);

		checkCudaErrors(cudaFree(LivePixels));

	}

	extern "C" void LaunchSavePNGKernel(
		unsigned char* ColorBytes,
		float3* Pixels,
		uint2 Resolution)
	{

		unsigned int BlockSize = BLOCK_SIZE;
		unsigned int GridSize = (Resolution.x*Resolution.y + BlockSize - 1) / BlockSize;

		checkCudaErrors(cudaDeviceSynchronize());
		SavePNG<<<GridSize, BlockSize>>>(
			ColorBytes,
			Pixels,
			Resolution);
		checkCudaErrors(cudaDeviceSynchronize());

	}

}
