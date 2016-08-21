#ifndef SCENE_H
#define SCENE_H

#include "Geometry.h"

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
	
	HSphere* spheres;
	unsigned int numSpheres;

private:
	HSceneData sceneData;

};

#endif // SCENE_H
