#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

namespace HKernels
{

	__global__ void TestKernel(float* d_in, float* d_out)
	{
		int i = threadIdx.x;
		d_out[i] = d_in[i] + 15.0f;
	}

	extern "C"
	void LaunchKernel(float* d_in, float* d_out)
	{
		TestKernel<<<1, 10>>>(d_in, d_out);
	}

}
