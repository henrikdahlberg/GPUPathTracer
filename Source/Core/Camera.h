#ifndef CAMERA_H
#define CAMERA_H

#include <Core/Include.h>
#include <Utility/MathUtility.h>

enum HCameraMovement {
	FORWARD,
	BACKWARD,
	LEFT,
	RIGHT,
	UP,
	DOWN,
};

struct HCameraData {
	glm::uvec2 resolution;
	glm::vec2 FOV;

	glm::vec3 position;
	glm::vec3 forward;
	glm::vec3 up;
	glm::vec3 right;
	glm::vec3 worldUp;

	// TODO: Lens model in separate class
	float apertureRadius;
	float focalDistance;
};

class HCamera {
public:
	HCamera(const unsigned int width, const unsigned int height);
	virtual ~HCamera() {}

	HCameraData* GetCameraData()				{ return &cameraData; }
	glm::uvec2 GetResolution()					{ return cameraData.resolution; }
	const glm::vec3 GetPosition() const			{ return cameraData.position; }
	const glm::vec3 GetForward() const			{ return cameraData.forward; }
	const glm::vec3 GetUp() const				{ return cameraData.up; }
	const glm::vec3 GetRight() const			{ return cameraData.right; }
	const glm::vec3 GetWorldUp() const			{ return cameraData.worldUp; }
	glm::vec2 GetFOV()							{ return cameraData.FOV; }

	void SetResolution(const unsigned int width,
					   const unsigned int height);
	void SetPosition(const glm::vec3 position)	{ cameraData.position = position; }
	void SetForward(const glm::vec3 forward)	{ cameraData.forward = forward; }
	void SetUp(const glm::vec3 up)				{ cameraData.up = up; }
	void SetRight(const glm::vec3 right)		{ cameraData.right = right; }
	void SetWorldUp(const glm::vec3 worldUp)	{ cameraData.worldUp = worldUp; }
	void SetFOV(const float FOV);

	void InitDefaults();
	// TODO: controller class...
	void ProcessMovement(HCameraMovement direction, const float deltaTime);
	void ProcessMouseMovement(float xoffset, float yoffset);
	void ProcessMouseScroll(float yoffset);
	
private:
	void UpdateCameraVectors();

	HCameraData cameraData;
	float velocity;
	float yaw;
	float pitch;
	// TODO: controller class...
	float mouseSensitivity;

};

#endif // CAMERA_H
