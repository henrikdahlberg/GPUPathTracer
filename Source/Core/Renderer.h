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
	HRenderer(HCamera* camera);
	virtual ~HRenderer();

	/**
	 * TODO: Doc
	 */
	HImage* Render();
	
	void InitScene(HScene* scene);
	void Update(HCamera* camera);
	void Resize(HCameraData* cameraData);

	unsigned int passCounter;
	unsigned int fpsCounter;

	
private:
	bool bFirstRenderPass;

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
	void DeleteVBO(GLuint* buffer, cudaGraphicsResource* bufferResource);

	void InitGPUData(HCameraData* cameraData);
	void FreeGPUData();

	cudaGraphicsResource* bufferResource;
	HImage* image;
	HCameraData* cameraData;
	HRay* rays;

	// Used in path tracing iteration instead of recursion
	float3* accumulatedColor;
	float3* colorMask;

	HSceneData* sceneData; // Not working, storing HSphere* for now
	// Temporary Scene storage
	HSphere* spheres;
	unsigned int numSpheres;

};

#endif // RENDERER_H
