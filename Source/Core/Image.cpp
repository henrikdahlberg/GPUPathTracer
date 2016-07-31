#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "Utility/External/stb_image_write.h"

#include "Image.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iostream>

#include "Kernels.h"

HImage::HImage()
{

}

HImage::HImage(unsigned int Width, unsigned int Height)
{
	Resolution.x = Width;
	Resolution.y = Height;
	NumPixels = Width*Height;
	Pixels = new float3[NumPixels];
}

HImage::HImage(uint2 Resolution)
{
	this->Resolution = Resolution;
	NumPixels = this->Resolution.x*this->Resolution.y;
	Pixels = new float3[NumPixels];
}

HImage::~HImage()
{
	delete[] Pixels;
	Pixels = nullptr;
}

void HImage::SavePNG(const std::string &Filename)
{

	unsigned char* ColorBytes = new unsigned char[3 * NumPixels];
	unsigned char* GPUColorBytes;
	size_t Size = 3 * NumPixels*sizeof(unsigned char);
	checkCudaErrors(cudaMalloc(&GPUColorBytes, Size));

	HKernels::LaunchSavePNGKernel(GPUColorBytes, GPUPixels, Resolution);

	checkCudaErrors(cudaMemcpy(ColorBytes, GPUColorBytes, Size, cudaMemcpyDeviceToHost));

	std::string FinalFilename = Filename + ".png";
	stbi_write_png(FinalFilename.c_str(), Resolution.x, Resolution.y, 3, ColorBytes, 3 * Resolution.x);

	delete[] ColorBytes;
	checkCudaErrors(cudaFree(GPUColorBytes));

}

void HImage::Resize(const unsigned int Width, const unsigned int Height)
{

	Resolution.x = Width;
	Resolution.y = Height;
	NumPixels = Width*Height;

}
