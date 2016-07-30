#include "Renderer.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <iostream>

HRenderer::HRenderer(HCameraData* CameraData)
{

	PassCounter = 0;
	FPSCounter = 0;
	bFirstRenderPass = true;
	this->CameraData = CameraData;
	Image = new HImage(CameraData->Resolution);
	InitCUDA();

}

HRenderer::HRenderer(HCamera* Camera)
{

	PassCounter = 0;
	FPSCounter = 0;
	bFirstRenderPass = true;
	this->CameraData = Camera->GetCameraData();
	Image = new HImage(CameraData->Resolution);
	InitCUDA();

}

HRenderer::~HRenderer()
{
	// TODO: Destructor, free CUDA pointers etc.
}

HImage* HRenderer::Render()
{

	if (bFirstRenderPass)
	{
		// TODO: Should be renamed to bReset or similar. Happens when camera is moved, reset GPU memory etc
		bFirstRenderPass = false;

	}

	++PassCounter;

	checkCudaErrors(cudaStreamCreate(&CUDAStream));
	checkCudaErrors(cudaGraphicsMapResources(1, &BufferResource, CUDAStream));

	// Launches CUDA kernel to modify OutImage pixels
	// Temporary test kernel for now to verify accumulation buffer
	HKernels::LaunchRenderKernel(
		Image->GPUPixels,
		AccumulationBuffer,
		CameraData,
		GPUCameraData,
		PassCounter,
		Rays);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &BufferResource, 0));

	checkCudaErrors(cudaStreamDestroy(CUDAStream));

	return Image;

}

void HRenderer::SetCameraData(HCameraData* CameraData)
{

	this->CameraData = CameraData;

}

void HRenderer::Reset()
{

	// TODO: Probably don't want to delete entire image object, just reset the pixels.
	//			Should update CameraData/GPUCameraData, reset AccumulationBuffer etc.
	if (Image != nullptr)
	{
		delete Image;
		Image = nullptr;
	}

	Image = new HImage(this->CameraData->Resolution);

}

void HRenderer::CreateVBO(GLuint* Buffer, cudaGraphicsResource** BufferResource, unsigned int BufferFlags)
{

	assert(Buffer);

	// Create buffer
	glGenBuffers(1, Buffer);
	glBindBuffer(GL_ARRAY_BUFFER, *Buffer);

	// Initialize buffer
	glBufferData(GL_ARRAY_BUFFER, Image->NumPixels * sizeof(float3), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Register VBO with CUDA and perform error checks
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(BufferResource, *Buffer, BufferFlags));
	SDK_CHECK_ERROR_GL();

}

void HRenderer::DeleteVBO(GLuint* Buffer, cudaGraphicsResource* BufferResource)
{

	// Unregister VBO with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(BufferResource));

	// Delete VBO
	glBindBuffer(1, *Buffer);
	glDeleteBuffers(1, Buffer);
	*Buffer = 0;

}

void HRenderer::InitCUDA()
{

	// Allocate memory on GPU for the accumulation buffer
	checkCudaErrors(cudaMalloc(&AccumulationBuffer, Image->NumPixels * sizeof(float3)));

	// Allocate memory on GPU for Camera data and copy over Camera data
	checkCudaErrors(cudaMalloc(&GPUCameraData, sizeof(HCameraData)));
	checkCudaErrors(cudaMemcpy(GPUCameraData, CameraData, sizeof(HCameraData), cudaMemcpyHostToDevice));

	// Allocate memory on GPU for rays
	checkCudaErrors(cudaMalloc(&Rays, Image->NumPixels * sizeof(HRay)));

	CreateVBO(&(Image->Buffer), &(this->BufferResource), cudaGraphicsRegisterFlagsNone);

	// Set up device synchronization stream
	checkCudaErrors(cudaStreamCreate(&CUDAStream));

	// Map graphics resource to CUDA
	checkCudaErrors(cudaGraphicsMapResources(1, &BufferResource, CUDAStream));

	// Set up access to mapped graphics resource through OutImage
	size_t NumBytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&(Image->GPUPixels), &NumBytes, BufferResource));

	// Unmap graphics resource, ensures synchronization
	checkCudaErrors(cudaGraphicsUnmapResources(1, &BufferResource, CUDAStream));

	// Clean up synchronization stream
	checkCudaErrors(cudaStreamDestroy(CUDAStream));

}