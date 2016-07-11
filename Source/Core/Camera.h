#ifndef CAMERA_H
#define CAMERA_H

#include <cuda_runtime.h>
#include "Core/Geometry.h"
#include "Utility/MathUtility.h"

struct CameraData
{
	float2 Resolution;
	float2 FOV;
	Vector3Df Position;
	Vector3Df Forward;
	Vector3Df Up;

	// TODO: Lens model in separate class
	float ApertureRadius;
	float FocalDistance;
};

class Camera
{
public:
	Camera();
	virtual ~Camera();
	
	/**
	 * Set camera resolution.
	 * @param Width		Width in pixels.
	 * @param Height	Height in pixels.
	 */
	void SetResolution(const float Width, const float Height);

	/**
	 * Set camera horizontal field of view and calculate
	 * vertical field of view based on resolution.
	 * @param NewFOV	Horizontal field of view in degrees.
	 */
	void SetFOV(const float NewFOV);

	float2 GetResolution() { return Data.Resolution; }
	float2 GetFOV() { return Data.FOV; }
	CameraData GetCameraData() { return Data; }
protected:

private:
	//TODO Make Transform component
	float Yaw;
	float Pitch;
	float Roll;

	CameraData Data;
};

#endif // CAMERA_H