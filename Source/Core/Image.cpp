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

HImage::HImage(unsigned int width, unsigned int height)
{
	resolution.x = width;
	resolution.y = height;
	numPixels = width*height;
}

HImage::HImage(uint2 resolution)
{
	this->resolution = resolution;
	numPixels = this->resolution.x*this->resolution.y;
}

HImage::~HImage()
{

}

void HImage::SavePNG(const std::string &filename)
{

	unsigned char* colorBytes = new unsigned char[3 * numPixels];
	unsigned char* GPUColorBytes;
	size_t size = 3 * numPixels*sizeof(unsigned char);
	checkCudaErrors(cudaMalloc(&GPUColorBytes, size));

	HKernels::LaunchSavePNGKernel(GPUColorBytes, pixels, resolution);

	checkCudaErrors(cudaMemcpy(colorBytes, GPUColorBytes, size, cudaMemcpyDeviceToHost));

	std::string finalFilename = filename + ".png";
	stbi_write_png(finalFilename.c_str(), resolution.x, resolution.y, 3, colorBytes, 3 * resolution.x);

	delete[] colorBytes;
	checkCudaErrors(cudaFree(GPUColorBytes));

}

void HImage::Resize(const unsigned int width, const unsigned int height)
{

	resolution.x = width;
	resolution.y = height;
	numPixels = width*height;

}
