#ifndef RENDERER_H
#define RENDERER_H

#include "GL/glew.h"

#include "Scene.h"
#include "Camera.h"
#include "Image.h"
#include "Kernels.h"

class HRenderer
{
public:
	HRenderer(HCameraData* CameraData);
	HRenderer(HCamera* Camera);
	virtual ~HRenderer();

	/**
	 * TODO: Doc
	 */
	HImage* Render();
	
	void SetCameraData(HCameraData* CameraData);
	void PrepareScene(HScene* Scene);
	void Reset();

	unsigned int PassCounter;
	unsigned int FPSCounter;

protected:
	/**
	 * Initializes OpenGL Vertex Buffer Object and registers it for access by CUDA.
	 *
	 * @param VBO
	 * @param VBOResource
	 * @param VBOFlags
	 */
	void CreateVBO(GLuint* Buffer, cudaGraphicsResource** BufferResource, unsigned int BufferFlags);

	/**
	 * TODO: Doc
	 *
	 * @param VBO
	 * @param VBOResource
	 */
	void DeleteVBO(GLuint* Buffer, cudaGraphicsResource* BufferResource);

private:
	bool bFirstRenderPass;

	void InitCUDA();
	cudaGraphicsResource* BufferResource;
	cudaStream_t CUDAStream;

	float3* AccumulationBuffer;
	HImage* Image;
	HCameraData* CameraData;
	HCameraData* GPUCameraData;
	HRay* Rays;
};

#endif // RENDERER_H
