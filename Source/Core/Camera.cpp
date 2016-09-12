#include <Core/Camera.h>

using namespace HMathUtility;

HCamera::HCamera(const unsigned int width, const unsigned int height) {
	// TODO: Pass scene parameters to set up initial position
	InitDefaults();
	SetResolution(width, height);
}

void HCamera::SetResolution(const unsigned int width, const unsigned int height) {
	cameraData.resolution = glm::uvec2(width, height);
	SetFOV(cameraData.FOV.x);
}

void HCamera::SetFOV(const float FOV) {
	cameraData.FOV.x = FOV;
	cameraData.FOV.y = RadToDeg(atan(tan(DegToRad(FOV)*0.5f)*((float)cameraData.resolution.y / (float)cameraData.resolution.x))*2.0f);
}

void HCamera::InitDefaults() {
	yaw = -90.0f;
	//pitch = 0.0f;
	pitch = -7.5f;
	cameraData.position = glm::vec3(0.00286f, 0.48820f, 1.9306f); // Cornell box camera position
	cameraData.worldUp = glm::vec3(0.0f, 1.0f, 0.0f);

	UpdateCameraVectors();
	cameraData.apertureRadius = 0.015f;
	//cameraData.focalDistance = 2.1f;
	cameraData.focalDistance = 1.9f;
	//cameraData.FOV.x = 39.0f; //Cornell
	cameraData.FOV.x = 35.0f;
	velocity = 3.0f;
	mouseSensitivity = 0.25f;
}

void HCamera::ProcessMovement(HCameraMovement direction, const float deltaTime) {
	float distance = velocity*deltaTime;
	switch (direction) {
	case FORWARD:
		cameraData.position += cameraData.forward * distance;	break;
	case BACKWARD:
		cameraData.position -= cameraData.forward * distance;	break;
	case LEFT:
		cameraData.position -= cameraData.right * distance;		break;
	case RIGHT:
		cameraData.position += cameraData.right * distance;		break;
	case UP:
		cameraData.position += cameraData.worldUp * distance;	break;
	case DOWN:
		cameraData.position -= cameraData.worldUp * distance;	break;
	}
}

void HCamera::ProcessMouseMovement(float xoffset, float yoffset) {

	yaw = fmod(yaw + xoffset*mouseSensitivity, 360.0f);
	pitch -= yoffset*mouseSensitivity;

	if (pitch > 89.0f) { pitch = 89.0f; }
	if (pitch < -89.0f) { pitch = -89.0f; }

	UpdateCameraVectors();
}

void HCamera::ProcessMouseScroll(float yoffset) {
	velocity = fmax(velocity + 0.25f*yoffset,0.0f);
}

void HCamera::UpdateCameraVectors() {
	glm::vec3 newForward;
	newForward.x = cos(glm::radians(this->yaw)) * cos(glm::radians(this->pitch));
	newForward.y = sin(glm::radians(this->pitch));
	newForward.z = sin(glm::radians(this->yaw)) * cos(glm::radians(this->pitch));

	cameraData.forward = normalize(newForward);
	cameraData.right = normalize(cross(cameraData.forward, cameraData.worldUp));
	cameraData.up = normalize(cross(cameraData.right, cameraData.forward));
}
