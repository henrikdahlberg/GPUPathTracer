#ifndef CAMERA_H
#define CAMERA_H

#include <Core/Include.h>
#include <Utility/MathUtility.h>

struct HCameraData {
	glm::uvec2 resolution;
	glm::vec2 FOV;
	glm::vec3 position;
	glm::vec3 forward;
	glm::vec3 up;

	// TODO: Lens model in separate class
	float apertureRadius;
	float focalDistance;
	float speed;
};

class HCamera {
public:
	HCamera(const unsigned int width, const unsigned int height);
	virtual ~HCamera() {}

	glm::mat3 GetOrientation() const;
	HCameraData* GetCameraData() { return &cameraData; }
	glm::uvec2 GetResolution() { return cameraData.resolution; }
	const glm::vec3 GetPosition() const { return cameraData.position; }
	const glm::vec3 GetForward() const { return cameraData.forward; }
	const glm::vec3 GetUp() const {	return cameraData.up; }
	float GetSpeed() const { return cameraData.speed; }
	glm::vec2 GetFOV() { return cameraData.FOV;	}
	glm::mat4 GetCameraToWorld() const;
	glm::mat4 GetWorldToCamera() const;

	void SetResolution(const unsigned int width, const unsigned int height);
	void SetPosition(const glm::vec3 position) { cameraData.position = position; }
	void SetForward(const glm::vec3 forward) { cameraData.forward = forward; }
	void SetUp(const glm::vec3 up) { cameraData.up = up; }
	void SetSpeed(float speed) { cameraData.speed = speed; }
	void SetFOV(const float FOV);
	void SetCameraToWorld(const glm::mat4 m);
	void SetWorldToCamera(const glm::mat4 m);

	void InitDefaults();

private:
	//TODO Make Transform component

	HCameraData cameraData;
};

#endif // CAMERA_H
