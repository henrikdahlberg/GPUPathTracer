#ifndef IMAGE_H
#define IMAGE_H

#include "GL/glew.h"
#include <string>

#include "Geometry.h"

class HImage
{
public:
	HImage();
	HImage(unsigned int Width, unsigned int Height);
	HImage(uint2 Resolution);
	virtual ~HImage();

	/**
	 * Saves current rendered image to .png file
	 * 
	 * TODO: If in the future we want to stop rendering and save .png
	 * after, we need to handle memory allocation of Pixels to GPU again.
	 * Ideally I would like to be able to access Pixels or the GL buffer
	 * directly on the CPU side somehow.
	 *
	 * @param Filename		Name of file to be saved.
	 */
	void SavePNG(const std::string &Filename);

	/**
	 * TODO: Doc
	 */
	void Resize(const unsigned int Width, const unsigned int Height);

	uint2 Resolution;
	GLuint Buffer;
	float3* Pixels;
	float3* GPUPixels;
	unsigned int NumPixels;

	// TODO: Save HDR image format from Accumulation buffer
};


#endif // IMAGE_H
