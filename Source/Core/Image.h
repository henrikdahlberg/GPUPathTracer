#ifndef IMAGE_H
#define IMAGE_H

#include <cuda.h>
#include <GL/glew.h>
#include <string>

#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif // GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <Core/Geometry.h>

class HImage
{
public:
	HImage();
	HImage(unsigned int width, unsigned int height);
	HImage(glm::uvec2 resolution);
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

	glm::uvec2 resolution;
	GLuint buffer;
	glm::vec3* pixels;
	glm::vec3* accumulationBuffer;
	unsigned int numPixels;

	// TODO: Save HDR image format from Accumulation buffer
};


#endif // IMAGE_H
