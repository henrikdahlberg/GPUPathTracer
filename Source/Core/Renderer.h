#ifndef RENDERER_H
#define RENDERER_H

#include <cuda.h>
#include <GL/glew.h>

#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif // GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <Core/Scene.h>
#include <Core/Camera.h>
#include <Core/Image.h>
#include <Core/Kernels.h>

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
	glm::vec3* accumulatedColor;
	glm::vec3* colorMask;

	HSceneData* sceneData; // Not working, storing HSphere* for now
	// Temporary Scene storage
	HSphere* spheres;
	unsigned int numSpheres;
	HTriangle* triangles;
	unsigned int numTriangles;

};

#endif // RENDERER_H
