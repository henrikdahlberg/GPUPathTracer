#ifndef CAMERA_H
#define CAMERA_H

#include <cuda_runtime.h>

#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif // GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <Utility/MathUtility.h>

struct HCameraData
{
	glm::uvec2 resolution;
	glm::vec2 FOV;
	glm::vec3 position;
	glm::vec3 view;
	glm::vec3 up;

	// TODO: Lens model in separate class
	float apertureRadius;
	float focalDistance;
};

class HCamera
{
public:
	HCamera(const unsigned int width, const unsigned int height);
	virtual ~HCamera();
	
	/**
	 * Set camera resolution.
	 * @param width		Width in pixels.
	 * @param height	Height in pixels.
	 */
	void SetResolution(const unsigned int width, const unsigned int height);

	/**
	 * Set camera horizontal field of view and calculate
	 * vertical field of view based on resolution.
	 * @param FOV	Horizontal field of view in degrees.
	 */
	void SetFOV(const float FOV);

	glm::uvec2 GetResolution() { return cameraData.resolution; }
	glm::vec2 GetFOV() { return cameraData.FOV; }
	HCameraData* GetCameraData() { return &cameraData; }

	void SetPosition(glm::vec3 position);

protected:
	void SetCameraData();

private:
	//TODO Make Transform component
	float yaw;
	float pitch;
	float roll;

	HCameraData cameraData;
};

#endif // CAMERA_H
