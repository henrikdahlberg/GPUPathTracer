#include "Renderer.h"

HRenderer::HRenderer(HCameraData* CameraData)
{
	bFirstRenderCall = true;
	this->CameraData = CameraData;
	OutImage = new HImage(CameraData->Resolution);
}

HRenderer::HRenderer(HCamera* Camera)
{
	bFirstRenderCall = true;
	this->CameraData = Camera->GetCameraData();
	OutImage = new HImage(CameraData->Resolution);
}

HRenderer::~HRenderer() {}

HImage* HRenderer::Render()
{
	if (bFirstRenderCall)
	{
		bFirstRenderCall = false;
	}

	return OutImage;
}

void HRenderer::SetCameraData(HCameraData* CameraData)
{
	this->CameraData = CameraData;
}

void HRenderer::Reset()
{
	if (OutImage != nullptr)
	{
		delete OutImage;
		OutImage = nullptr;
	}

	OutImage = new HImage(CameraData->Resolution);
}

void HRenderer::TestRunKernel(float* d_in, float* d_out)
{
	HKernels::LaunchKernel(d_in, d_out);
}

void HRenderer::CreateVBO(GLuint* VBO, cudaGraphicsResource** VBOResource, unsigned int VBOFlags)
{
	assert(VBO);

	// Create VBO
	glGenBuffers(1, VBO);
	glBindBuffer(GL_ARRAY_BUFFER, *VBO);

	// Initialize VBO
	unsigned int VBOSize = OutImage->Resolution.x * OutImage->Resolution.y * sizeof(Vector3Df);
	glBufferData(GL_ARRAY_BUFFER, VBOSize, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Register VBO with CUDA and perform error checks
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(VBOResource, *VBO, VBOFlags));
	SDK_CHECK_ERROR_GL();
}

void HRenderer::DeleteVBO(GLuint* VBO, cudaGraphicsResource* VBOResource)
{
	// Unregister VBO with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(VBOResource));
	glBindBuffer(1, *VBO);
	glDeleteBuffers(1, VBO);
	*VBO = 0;
}
