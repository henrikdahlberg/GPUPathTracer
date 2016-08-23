#ifndef FILELOADER_H
#define FILELOADER_H

#include <stdio.h>
#include <vector>

#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif // GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <Shapes/Triangle.h>

class HFileLoader
{
public:
	HFileLoader()
	{

	}

	~HFileLoader() {}

	bool LoadOBJ(const char* filePath,
				 HTriangleMesh &mesh);

private:


};

#endif // FILELOADER_H