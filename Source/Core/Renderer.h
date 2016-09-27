#ifndef RENDERER_H
#define RENDERER_H

#include <Core/Include.h>

#include <Core/Scene.h>
#include <Core/Camera.h>
#include <Core/Image.h>
#include <Core/Kernels.h>

class HRenderer {
public:
	HRenderer(HCamera* camera);
	virtual ~HRenderer();

	HImage* Render();

	void InitScene(HScene* scene);
	void Reset(HCamera* camera);
	void Resize(HCameraData* cameraData);

	unsigned int passCounter;

private:
	bool bFirstRenderPass;

	void CreateVBO(GLuint* Buffer, cudaGraphicsResource** BufferResource, unsigned int BufferFlags);
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

	BVH bvh;
};

#endif // RENDERER_H
