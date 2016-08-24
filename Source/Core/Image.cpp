#include <Core/Include.h>

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <Utility/External/stb_image_write.h>
#endif // STB_IMAGE_WRITE_IMPLEMENTATION

#include <Core/Image.h>
#include <Core/Kernels.h>

HImage::HImage(unsigned int width, unsigned int height) {
	resolution.x = width;
	resolution.y = height;
	numPixels = width*height;
}

HImage::HImage(glm::uvec2 resolution) {
	this->resolution = resolution;
	numPixels = this->resolution.x*this->resolution.y;
}

void HImage::SavePNG(const std::string &filename) {

	unsigned char* colorBytes = new unsigned char[3 * numPixels];
	unsigned char* GPUColorBytes;
	size_t size = 3 * numPixels*sizeof(unsigned char);
	checkCudaErrors(cudaMalloc(&GPUColorBytes, size));

	HKernels::LaunchSavePNGKernel(GPUColorBytes, pixels, resolution.x, resolution.y);

	checkCudaErrors(cudaMemcpy(colorBytes, GPUColorBytes, size, cudaMemcpyDeviceToHost));

	std::string finalFilename = filename + ".png";
	stbi_write_png(finalFilename.c_str(), resolution.x, resolution.y, 3, colorBytes, 3 * resolution.x);

	delete[] colorBytes;
	checkCudaErrors(cudaFree(GPUColorBytes));
}

void HImage::Resize(const unsigned int width, const unsigned int height) {
	resolution.x = width;
	resolution.y = height;
	numPixels = width*height;
}
