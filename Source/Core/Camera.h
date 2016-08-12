#ifndef CAMERA_H
#define CAMERA_H

#include <cuda_runtime.h>

#include "Utility/MathUtility.h"

struct HCameraData
{
	uint2 Resolution;
	float2 FOV;
	float3 Position;
	float3 View;
	float3 Up;

	// TODO: Lens model in separate class
	float ApertureRadius;
	float FocalDistance;
};

class HCamera
{
public:
	HCamera(const unsigned int Width, const unsigned int Height);
	virtual ~HCamera();
	
	/**
	 * Set camera resolution.
	 * @param Width		Width in pixels.
	 * @param Height	Height in pixels.
	 */
	void SetResolution(const unsigned int Width, const unsigned int Height);

	/**
	 * Set camera horizontal field of view and calculate
	 * vertical field of view based on resolution.
	 * @param NewFOV	Horizontal field of view in degrees.
	 */
	void SetFOV(const float NewFOV);

	uint2 GetResolution() { return CameraData.Resolution; }
	float2 GetFOV() { return CameraData.FOV; }
	HCameraData* GetCameraData() { return &CameraData; }

	void SetPosition(float3 NewPosition);

protected:
	void SetCameraData();

private:
	//TODO Make Transform component
	float Yaw;
	float Pitch;
	float Roll;

	HCameraData CameraData;
};

#endif // CAMERA_H
