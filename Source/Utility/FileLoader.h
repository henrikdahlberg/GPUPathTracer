#ifndef FILELOADER_H
#define FILELOADER_H

#include <Core/Include.h>
#include <Shapes/Triangle.h>

class HFileLoader {
public:
	HFileLoader() {}
	~HFileLoader() {}

	bool LoadOBJ(const char* filePath,
				 HTriangleMesh &mesh);

private:

};

#endif // FILELOADER_H