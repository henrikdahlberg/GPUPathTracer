#include "Camera.h"

HCamera::HCamera()
{
	// TODO: Take window size as argument

	CameraData.Position = make_float3(0.0f, 0.0f, 0.0f);
	Yaw = 0.0f;
	Pitch = 0.0f;
	Roll = 0.0f;
	CameraData.ApertureRadius = 0.05f;
	CameraData.FocalDistance = 5.0f;

	SetResolution(1280, 720);
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
