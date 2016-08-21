#ifndef CAMERA_H
#define CAMERA_H

#include <cuda_runtime.h>

#include "Utility/MathUtility.h"

struct HCameraData
{
	uint2 resolution;
	float2 FOV;
	float3 position;
	float3 view;
	float3 up;

	// TODO: Lens model in separate class
	float apertureRadius;
	float focalDistance;
};

class HCamera
{
public:
	HCamera(const unsigned int width, const unsigned int height);
	virtual ~HCamera();
	
	/**
	 * Set camera resolution.
	 * @param width		Width in pixels.
	 * @param height	Height in pixels.
	 */
	void SetResolution(const unsigned int width, const unsigned int height);

	/**
	 * Set camera horizontal field of view and calculate
	 * vertical field of view based on resolution.
	 * @param FOV	Horizontal field of view in degrees.
	 */
	void SetFOV(const float FOV);

	uint2 GetResolution() { return cameraData.resolution; }
	float2 GetFOV() { return cameraData.FOV; }
	HCameraData* GetCameraData() { return &cameraData; }

	void SetPosition(float3 position);

protected:
	void SetCameraData();

private:
	//TODO Make Transform component
	float yaw;
	float pitch;
	float roll;

	HCameraData cameraData;
};

#endif // CAMERA_H
