#include "Camera.h"

#include <math.h>

HCamera::HCamera(const unsigned int width, const unsigned int height)
{
	// TODO: Pass scene parameters to set up initial position

	cameraData.position = make_float3(0.0f, 0.5f, 1.0f);
	cameraData.view = make_float3(0.0f, 0.0f, -1.0f);
	cameraData.up = make_float3(0.0f, 1.0f, 0.0f);
	yaw = 0.0f;
	pitch = 0.0f;
	roll = 0.0f;
	cameraData.apertureRadius = 0.05f;
	cameraData.focalDistance = 2.5f;
	cameraData.FOV.x = 90.0f;

	SetResolution(width, height);
}

HCamera::~HCamera() {}

void HCamera::SetResolution(const unsigned int width, const unsigned int height)
{
	cameraData.resolution = make_uint2(width, height);
	SetFOV(cameraData.FOV.x);
}

void HCamera::SetFOV(const float FOV)
{
	cameraData.FOV.x = FOV;
	cameraData.FOV.y = HMathUtility::RadToDeg(atan(tan(HMathUtility::DegToRad(FOV)*0.5f)*((float)cameraData.resolution.y / (float)cameraData.resolution.x))*2.0f);
}

void HCamera::SetPosition(float3 position)
{

	cameraData.position = position;

}


