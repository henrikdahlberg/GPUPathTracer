#include <math.h>
#include <cuda.h>

#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif // GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <Core/Camera.h>

HCamera::HCamera(const unsigned int width, const unsigned int height)
{
	// TODO: Pass scene parameters to set up initial position

	cameraData.position = glm::vec3(0.3f, -1.4f, 1.4f);
	cameraData.view = glm::vec3(0.0f, 1.0f, -0.6f);
	cameraData.up = glm::vec3(0.0f, 1.0f, 0.0f);
	yaw = 0.0f;
	pitch = 0.0f;
	roll = 0.0f;
	cameraData.apertureRadius = 0.08f;
	cameraData.focalDistance = 1.5f;
	cameraData.FOV.x = 75.0f;

	SetResolution(width, height);
}

HCamera::~HCamera() {}

void HCamera::SetResolution(const unsigned int width, const unsigned int height)
{
	cameraData.resolution = glm::uvec2(width, height);
	SetFOV(cameraData.FOV.x);
}

void HCamera::SetFOV(const float FOV)
{
	cameraData.FOV.x = FOV;
	cameraData.FOV.y = HMathUtility::RadToDeg(atan(tan(HMathUtility::DegToRad(FOV)*0.5f)*((float)cameraData.resolution.y / (float)cameraData.resolution.x))*2.0f);
}

void HCamera::SetPosition(glm::vec3 position)
{

	cameraData.position = position;

}


