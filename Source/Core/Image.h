#ifndef IMAGE_H
#define IMAGE_H

#include "Geometry.h"

class HImage
{
public:
	HImage();
	HImage(unsigned int Width, unsigned int Height);
	HImage(float2 Resolution);
	virtual ~HImage();

	float2 Resolution;
	Vector3Df* Pixels;
	unsigned int NumPixels;
};


#endif // IMAGE_H
