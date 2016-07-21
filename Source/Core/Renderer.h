#ifndef RENDERER_H
#define RENDERER_H

#include "GL/glew.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

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

	//temp
	void TestRunKernel(float* d_in, float* d_out);

protected:
	/**
	 * TODO: Doc
	 * @param VBO
	 * @param VBOResource
	 * @param VBOFlags
	 */
	void CreateVBO(GLuint* VBO, cudaGraphicsResource** VBOResource, unsigned int VBOFlags);

	/**
	 * TODO: Doc
	 * @param VBO
	 * @param VBOResource
	 */
	void DeleteVBO(GLuint* VBO, cudaGraphicsResource* VBOResource);

private:
	bool bFirstRenderCall;

	GLuint VBO;

	Vector3Df* AccumulationBuffer;

	HImage* OutImage;
	HCameraData* CameraData;
};

#endif // RENDERER_H
