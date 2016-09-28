#include <Core/Renderer.h>
#include <Core/BVHConstruction.h>

HRenderer::HRenderer(HCamera* camera) {
	passCounter = 0;
	bFirstRenderPass = true;
	image = new HImage(camera->GetCameraData()->resolution);
	InitGPUData(camera->GetCameraData());
}

HRenderer::~HRenderer() {
	// TODO: Destructor, free CUDA pointers, delete Image, CameraData etc.
	FreeGPUData();
}

HImage* HRenderer::Render() {

	if (bFirstRenderPass) {
		// TODO: Should be renamed to bReset or similar. Happens when camera is moved, reset GPU memory etc
		// TODO: perhaps not even needed
		bFirstRenderPass = false;
	}

	++passCounter;

	cudaStream_t CUDAStream;
	checkCudaErrors(cudaStreamCreate(&CUDAStream));
	checkCudaErrors(cudaGraphicsMapResources(1, &bufferResource, CUDAStream));

	// Launches CUDA kernel to modify Image pixels
	HKernels::LaunchRenderKernel(image,
								 accumulatedColor,
								 colorMask,
								 cameraData,
								 passCounter,
								 rays,
								 spheres,
								 numSpheres,
								 bvh.BVHNodes,
								 triangles,
								 numTriangles);

	if (passCounter % 1000 == 0) {
		image->SavePNG("Images/Autosave");
	}

	checkCudaErrors(cudaGraphicsUnmapResources(1, &bufferResource, 0));
	checkCudaErrors(cudaStreamDestroy(CUDAStream));

	return image;
}

void HRenderer::InitScene(HScene* scene) {

	numSpheres = scene->numSpheres;
	checkCudaErrors(cudaMalloc(&spheres, numSpheres*sizeof(HSphere)));
	checkCudaErrors(cudaMemcpy(spheres, scene->spheres, numSpheres*sizeof(HSphere), cudaMemcpyHostToDevice));

	numTriangles = scene->numTriangles;
	checkCudaErrors(cudaMalloc(&triangles, numTriangles*sizeof(HTriangle)));
	checkCudaErrors(cudaMemcpy(triangles, scene->triangles.data(), numTriangles*sizeof(HTriangle), cudaMemcpyHostToDevice));

	BuildBVH(bvh, triangles, numTriangles, scene->sceneBounds);

}

void HRenderer::Reset(HCamera* camera) {

	FreeGPUData();

	image->Resize(camera->GetCameraData()->resolution.x,
				  camera->GetCameraData()->resolution.y);

	passCounter = 0;

	InitGPUData(camera->GetCameraData());
}

void HRenderer::Resize(HCameraData* cameraData) {

	FreeGPUData();

	image->Resize(cameraData->resolution.x, cameraData->resolution.y);

	passCounter = 0;

	InitGPUData(cameraData);
}

void HRenderer::CreateVBO(GLuint* buffer,
						  cudaGraphicsResource** bufferResource,
						  unsigned int flags) {

	assert(buffer);

	// Create buffer
	glGenBuffers(1, buffer);
	glBindBuffer(GL_ARRAY_BUFFER, *buffer);

	// Initialize buffer
	glBufferData(GL_ARRAY_BUFFER, image->numPixels * sizeof(glm::vec3), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Register VBO with CUDA and perform error checks
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(bufferResource, *buffer, flags));
}

void HRenderer::DeleteVBO(GLuint* buffer, cudaGraphicsResource* bufferResource) {

	// Unregister VBO with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(bufferResource));

	// Delete VBO
	glBindBuffer(GL_ARRAY_BUFFER, *buffer);
	glDeleteBuffers(1, buffer);
	*buffer = 0;
}

void HRenderer::InitGPUData(HCameraData* cameraData) {

	// Allocate memory on GPU for the accumulation buffer
	checkCudaErrors(cudaMalloc(&(image->accumulationBuffer), image->numPixels * sizeof(glm::vec3)));

	// Allocate memory on GPU for Camera data and copy over Camera data
	checkCudaErrors(cudaMalloc(&(this->cameraData), sizeof(HCameraData)));
	checkCudaErrors(cudaMemcpy(this->cameraData, cameraData, sizeof(HCameraData), cudaMemcpyHostToDevice));

	// Allocate memory on GPU for rays
	checkCudaErrors(cudaMalloc(&rays, image->numPixels * sizeof(HRay)));

	// Allocate memory on GPU for path tracing iteration
	checkCudaErrors(cudaMalloc(&accumulatedColor, image->numPixels * sizeof(glm::vec3)));
	checkCudaErrors(cudaMalloc(&colorMask, image->numPixels * sizeof(glm::vec3)));

	CreateVBO(&(image->buffer), &(this->bufferResource), cudaGraphicsRegisterFlagsNone);

	// Set up device synchronization stream
	cudaStream_t CUDAStream;
	checkCudaErrors(cudaStreamCreate(&CUDAStream));
	// Map graphics resource to CUDA
	checkCudaErrors(cudaGraphicsMapResources(1, &bufferResource, CUDAStream));
	// Set up access to mapped graphics resource through Image
	size_t NumBytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&(image->pixels), &NumBytes, bufferResource));
	// Unmap graphics resource, ensures synchronization
	checkCudaErrors(cudaGraphicsUnmapResources(1, &bufferResource, CUDAStream));
	// Clean up synchronization stream
	checkCudaErrors(cudaStreamDestroy(CUDAStream));
}

void HRenderer::FreeGPUData() {
	DeleteVBO(&(image->buffer), this->bufferResource);
	checkCudaErrors(cudaFree(image->accumulationBuffer));
	checkCudaErrors(cudaFree(cameraData));
	checkCudaErrors(cudaFree(rays));
	checkCudaErrors(cudaFree(accumulatedColor));
	checkCudaErrors(cudaFree(colorMask));
	// TODO: Free scene data
}