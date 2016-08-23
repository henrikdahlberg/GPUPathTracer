#ifndef SCENE_H
#define SCENE_H

#include <Core/Geometry.h>
#include <Shapes/Sphere.h>
#include <Utility/FileLoader.h>

struct HSceneData
{
	HSphere* spheres;
	unsigned int numSpheres;
};

class HScene
{
public:
	HScene();
	virtual ~HScene();

	// TODO: implement
	void LoadSceneFile();

	HSceneData* GetSceneData() { return &sceneData; }
	
	HTriangle* triangles;
	unsigned int numTriangles;

	HSphere* spheres;
	unsigned int numSpheres;

private:
	HSceneData sceneData;
	HFileLoader fileLoader;

};

#endif // SCENE_H
