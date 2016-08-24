#ifndef IMAGE_H
#define IMAGE_H

#include <Core/Include.h>
#include <Core/Geometry.h>

class HImage
{
public:
	HImage() {}
	HImage(unsigned int width, unsigned int height);
	HImage(glm::uvec2 resolution);
	virtual ~HImage() {}

	void SavePNG(const std::string &filename);
	void Resize(const unsigned int width, const unsigned int height);

	glm::uvec2 resolution;
	GLuint buffer;
	glm::vec3* pixels;
	glm::vec3* accumulationBuffer;
	unsigned int numPixels;

	// TODO: Save HDR image format from Accumulation buffer
};


#endif // IMAGE_H
