#include "Renderer.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <iostream>

HRenderer::HRenderer(HCamera* Camera)
{

	PassCounter = 0;
	FPSCounter = 0;
	bFirstRenderPass = true;
	Image = new HImage(Camera->GetCameraData()->Resolution);
	InitGPUData(Camera->GetCameraData());

}

HRenderer::~HRenderer()
{
	// TODO: Destructor, free CUDA pointers, delete Image, CameraData etc.
	FreeGPUData();

}

HImage* HRenderer::Render()
{

	if (bFirstRenderPass)
	{
		// TODO: Should be renamed to bReset or similar. Happens when camera is moved, reset GPU memory etc
		// TODO: perhaps not even needed
		bFirstRenderPass = false;

	}

	++PassCounter;

	cudaStream_t CUDAStream;
	checkCudaErrors(cudaStreamCreate(&CUDAStream));
	checkCudaErrors(cudaGraphicsMapResources(1, &BufferResource, CUDAStream));

	// Launches CUDA kernel to modify Image pixels
	HKernels::LaunchRenderKernel(
		Image,
		AccumulatedColor,
		ColorMask,
		CameraData,
		PassCounter,
		Rays,
		Spheres,
		NumSpheres);
	
	if (PassCounter == 10000)
	{
		//Image->SavePNG("Images/");
	}

	checkCudaErrors(cudaGraphicsUnmapResources(1, &BufferResource, 0));
	checkCudaErrors(cudaStreamDestroy(CUDAStream));

	return Image;

}

void HRenderer::InitScene(HScene* Scene)
{

	NumSpheres = Scene->NumSpheres;
	checkCudaErrors(cudaMalloc(&Spheres, NumSpheres*sizeof(HSphere)));
	checkCudaErrors(cudaMemcpy(Spheres, Scene->Spheres, NumSpheres*sizeof(HSphere), cudaMemcpyHostToDevice));

	
	// TODO: Unified memory to skip this tedious deep copy
	/*HSphere* TempSpheres;
	checkCudaErrors(cudaMalloc(&SceneData, sizeof(HSceneData)));
	checkCudaErrors(cudaMalloc(&TempSpheres, Scene->GetSceneData()->NumSpheres * sizeof(HSphere)));
	checkCudaErrors(cudaMemcpy(this->SceneData, Scene->GetSceneData(), sizeof(HSceneData), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(TempSpheres, Scene->GetSceneData()->Spheres, Scene->GetSceneData()->NumSpheres * sizeof(HSphere), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(&(SceneData->Spheres), &TempSpheres, sizeof(HSphere*), cudaMemcpyHostToDevice));*/

	// This unified memory snipped doesn't work, copies data but crashes when trying to access
	// Spheres pointer in SceneData struct on GPU
	/*checkCudaErrors(cudaMallocManaged(&SceneData, sizeof(HSceneData)));
	checkCudaErrors(cudaMemcpy(this->SceneData, Scene->GetSceneData(), sizeof(HSceneData), cudaMemcpyHostToDevice));*/

}

void HRenderer::Update(HCamera* Camera)
{

	FreeGPUData();

	Image->Resize(
		Camera->GetCameraData()->Resolution.x,
		Camera->GetCameraData()->Resolution.y);

	PassCounter = 0;

	InitGPUData(Camera->GetCameraData());

}

void HRenderer::Resize(HCameraData* CameraData)
{

	FreeGPUData();

	Image->Resize(CameraData->Resolution.x, CameraData->Resolution.y);

	PassCounter = 0;

	InitGPUData(CameraData);

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

}

void HRenderer::DeleteVBO(GLuint* Buffer, cudaGraphicsResource* BufferResource)
{

	// Unregister VBO with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(BufferResource));

	// Delete VBO
	glBindBuffer(GL_ARRAY_BUFFER, *Buffer);
	glDeleteBuffers(1, Buffer);
	*Buffer = 0;

}

void HRenderer::InitGPUData(HCameraData* CameraData)
{

	// Allocate memory on GPU for the accumulation buffer
	checkCudaErrors(cudaMalloc(&(Image->AccumulationBuffer), Image->NumPixels * sizeof(float3)));

	// Allocate memory on GPU for Camera data and copy over Camera data
	checkCudaErrors(cudaMalloc(&(this->CameraData), sizeof(HCameraData)));
	checkCudaErrors(cudaMemcpy(this->CameraData, CameraData, sizeof(HCameraData), cudaMemcpyHostToDevice));

	// Allocate memory on GPU for rays
	checkCudaErrors(cudaMalloc(&Rays, Image->NumPixels * sizeof(HRay)));

	// Allocate memory on GPU for path tracing iteration
	checkCudaErrors(cudaMalloc(&AccumulatedColor, Image->NumPixels * sizeof(float3)));
	checkCudaErrors(cudaMalloc(&ColorMask, Image->NumPixels * sizeof(float3)));

	CreateVBO(&(Image->Buffer), &(this->BufferResource), cudaGraphicsRegisterFlagsNone);

	// Set up device synchronization stream
	cudaStream_t CUDAStream;
	checkCudaErrors(cudaStreamCreate(&CUDAStream));

	// Map graphics resource to CUDA
	checkCudaErrors(cudaGraphicsMapResources(1, &BufferResource, CUDAStream));

	// Set up access to mapped graphics resource through Image
	size_t NumBytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&(Image->Pixels), &NumBytes, BufferResource));

	// Unmap graphics resource, ensures synchronization
	checkCudaErrors(cudaGraphicsUnmapResources(1, &BufferResource, CUDAStream));

	// Clean up synchronization stream
	checkCudaErrors(cudaStreamDestroy(CUDAStream));

}

void HRenderer::FreeGPUData()
{

	DeleteVBO(&(Image->Buffer), this->BufferResource);

	checkCudaErrors(cudaFree(Image->AccumulationBuffer));
	checkCudaErrors(cudaFree(CameraData));
	checkCudaErrors(cudaFree(Rays));
	checkCudaErrors(cudaFree(AccumulatedColor));
	checkCudaErrors(cudaFree(ColorMask));

	// TODO: Free scene data

}