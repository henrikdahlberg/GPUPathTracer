#include "Camera.h"

#include <math.h>

HCamera::HCamera(const unsigned int Width, const unsigned int Height)
{
	// TODO: Pass scene parameters to set up initial position

	CameraData.Position = make_float3(0.0f, 0.0f, 2.0f);
	CameraData.View = make_float3(0.0f, 0.0f, -1.0f);
	CameraData.Up = make_float3(0.0f, 1.0f, 0.0f);
	Yaw = 0.0f;
	Pitch = 0.0f;
	Roll = 0.0f;
	CameraData.ApertureRadius = 0.05f;
	CameraData.FocalDistance = 3.0f;
	CameraData.FOV.x = 90.0f;

	SetResolution(Width, Height);
}

HCamera::~HCamera() {}

void HCamera::SetResolution(const unsigned int Width, const unsigned int Height)
{
	CameraData.Resolution = make_uint2(Width, Height);
	SetFOV(CameraData.FOV.x);
}

void HCamera::SetFOV(const float NewFOV)
{
	CameraData.FOV.x = NewFOV;
	CameraData.FOV.y = HMathUtility::RadToDeg(atan(tan(HMathUtility::DegToRad(NewFOV)*0.5f)*((float)CameraData.Resolution.y / (float)CameraData.Resolution.x))*2.0f);
}
