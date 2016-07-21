#ifndef KERNELS_H
#define KERNELS_H

namespace HKernels
{

	// The 'extern "C"' declaration is necessary in order to call
	// CUDA kernels defined in .cu-files from .cpp files
	extern "C"
	void LaunchKernel(float* d_in, float* d_out);

}

#endif //KERNELS_H
