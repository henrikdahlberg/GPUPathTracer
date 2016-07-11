#include "Camera.h"

Camera::Camera()
{
	Data.Position = Vector3Df(0.0f, 0.0f, 0.0f);
	Yaw = 0.0f;
	Pitch = 0.0f;
	Roll = 0.0f;
	Data.ApertureRadius = 0.05f;
	Data.FocalDistance = 5.0f;

	Data.Resolution = make_float2(1280, 720);
	Data.FOV = make_float2(45, 45);
}

Camera::~Camera() {}

void Camera::SetResolution(const float Width, const float Height)
{
	Data.Resolution = make_float2(Width, Height);
	SetFOV(Data.FOV.x);
}

void Camera::SetFOV(const float NewFOV)
{
	Data.FOV.x = NewFOV;
	Data.FOV.y = MathUtility::RadToDeg(atan(tan(MathUtility::DegToRad(NewFOV)*0.5f)*(Data.Resolution.y / Data.Resolution.x))*2.0f);
}
