#ifndef IMAGE_H
#define IMAGE_H

#include "GL/glew.h"
#include <string>

#include "Geometry.h"

class HImage
{
public:
	HImage();
	HImage(unsigned int width, unsigned int height);
	HImage(uint2 resolution);
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
	void SavePNG(const std::string &filename);

	/**
	 * TODO: Doc
	 */
	void Resize(const unsigned int width, const unsigned int height);

	uint2 resolution;
	GLuint buffer;
	float3* pixels;
	float3* accumulationBuffer;
	unsigned int numPixels;

	// TODO: Save HDR image format from Accumulation buffer
};


#endif // IMAGE_H
