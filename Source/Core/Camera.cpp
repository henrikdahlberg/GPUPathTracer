#include "Camera.h"

HCamera::HCamera()
{
	CameraData.Position = Vector3Df(0.0f, 0.0f, 0.0f);
	Yaw = 0.0f;
	Pitch = 0.0f;
	Roll = 0.0f;
	CameraData.ApertureRadius = 0.05f;
	CameraData.FocalDistance = 5.0f;

	CameraData.Resolution = make_float2(1280, 720);
	CameraData.FOV = make_float2(45, 45);
}

HCamera::~HCamera() {}

void HCamera::SetResolution(const float Width, const float Height)
{
	CameraData.Resolution = make_float2(Width, Height);
	SetFOV(CameraData.FOV.x);
}

void HCamera::SetFOV(const float NewFOV)
{
	CameraData.FOV.x = NewFOV;
	CameraData.FOV.y = HMathUtility::RadToDeg(atan(tan(HMathUtility::DegToRad(NewFOV)*0.5f)*(CameraData.Resolution.y / CameraData.Resolution.x))*2.0f);
}
