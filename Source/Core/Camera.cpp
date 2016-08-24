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

glm::mat3 HCamera::GetOrientation() const {
	glm::mat3 m;
	m[2] = -normalize(cameraData.forward);
	m[0] = normalize(cross(cameraData.up, m[2]));
	m[1] = normalize(cross(m[2], m[0]));
	return m;
}

glm::mat4 HCamera::GetCameraToWorld() const {
	glm::mat3 o = GetOrientation();
	glm::mat4 m;
	m[0] = glm::vec4(o[0], 0.0f);
	m[1] = glm::vec4(o[1], 0.0f);
	m[2] = glm::vec4(o[2], 0.0f);
	m[3] = glm::vec4(cameraData.position, 1.0f);
	return m;
}

glm::mat4 HCamera::GetWorldToCamera() const {
	return glm::mat4();
}

void HCamera::SetCameraToWorld(const glm::mat4 m) {

}

void HCamera::SetWorldToCamera(const glm::mat4 m) {

}

void HCamera::InitDefaults() {
	cameraData.position = glm::vec3(0.0f, 0.5f, 1.5f);
	cameraData.forward = glm::vec3(0.0f, 0.0f, -1.0f);
	cameraData.up = glm::vec3(0.0f, 1.0f, 0.0f);
	cameraData.apertureRadius = 0.02f;
	cameraData.focalDistance = 1.5f;
	cameraData.FOV.x = 75.0f;
}

