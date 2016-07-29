#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include "Geometry.h"
#include "Camera.h"
#include "Utility/MathUtility.h"

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

	//////////////////////////////////////////////////////////////////////////
	// Global Kernels
	//////////////////////////////////////////////////////////////////////////
	__global__ void TestRenderKernel(
		float3* Pixels,
		float3* AccumulationBuffer,
		HCameraData* CameraData,
		unsigned int PassCounter)
	{
		
		int x = blockIdx.x*blockDim.x + threadIdx.x;
		int y = blockIdx.y*blockDim.y + threadIdx.y;

		// Global thread ID, used to perturb the random number generator seed
		int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

		int i = (CameraData->Resolution.y - y - 1)*CameraData->Resolution.x + x;

		// Initialize random number generator
		curandState RNGState;
		curand_init(TWHash(PassCounter) + threadId, 0, 0, &RNGState);

		// Random test color
		float3 TempColor = make_float3(curand_uniform(&RNGState), curand_uniform(&RNGState), curand_uniform(&RNGState));

		// Accumulate and average the color for each pass
		AccumulationBuffer[i] = (AccumulationBuffer[i] * (PassCounter - 1) + TempColor) / PassCounter;

		TempColor = AccumulationBuffer[i];

		// Make type conversion for OpenGL and perform gamma correction
		HColor Color;
		Color.Components = make_uchar4(
			(unsigned char)(powf(TempColor.x, 1 / 2.2f) * 255),
			(unsigned char)(powf(TempColor.y, 1 / 2.2f) * 255),
			(unsigned char)(powf(TempColor.z, 1 / 2.2f) * 255), 1);

		// Pass pixel coordinates and pixel color in OpenGL to output buffer
		Pixels[i] = make_float3(x, y, Color.Value);

	}

	__global__ void SavePNGKernel(
		unsigned char* ColorBytes,
		float3* Pixels,
		uint2 Resolution)
	{

		int x = blockIdx.x*blockDim.x + threadIdx.x;
		int y = blockIdx.y*blockDim.y + threadIdx.y;

		int i = (Resolution.y - y - 1)*Resolution.x + x;
		
		HColor Color;
		Color.Value = Pixels[i].z;

		ColorBytes[3 * i    ] = (unsigned char)Color.Components.x;
		ColorBytes[3 * i + 1] = (unsigned char)Color.Components.y;
		ColorBytes[3 * i + 2] = (unsigned char)Color.Components.z;

	}

	//////////////////////////////////////////////////////////////////////////
	// External CUDA access launch function
	//////////////////////////////////////////////////////////////////////////
	extern "C" void LaunchRenderKernel(
		float3* Pixels,
		float3* AccumulationBuffer,
		HCameraData* CameraData,
		HCameraData* GPUCameraData,
		unsigned int PassCounter)
	{

		const dim3 BlockSize(16, 16, 1);
		const dim3 GridSize(CameraData->Resolution.x / BlockSize.x, CameraData->Resolution.y / BlockSize.y, 1);

		TestRenderKernel<<<GridSize, BlockSize>>>(
			Pixels,
			AccumulationBuffer,
			GPUCameraData,
			PassCounter);

	}

	extern "C" void LaunchSavePNGKernel(
		unsigned char* ColorBytes,
		float3* Pixels,
		uint2 Resolution)
	{

		const dim3 BlockSize(16, 16, 1);
		const dim3 GridSize(Resolution.x / BlockSize.x, Resolution.y / BlockSize.y, 1);

		checkCudaErrors(cudaDeviceSynchronize());
		SavePNGKernel<<<GridSize, BlockSize>>>(
			ColorBytes,
			Pixels,
			Resolution);
		checkCudaErrors(cudaDeviceSynchronize());

	}

}
