#ifndef SCENE_H
#define SCENE_H

#include "Geometry.h"

struct HSceneData
{
	HSphere* Spheres;
	unsigned int NumSpheres;
};

class HScene
{
public:
	HScene();
	virtual ~HScene();

	// TODO: implement
	void LoadSceneFile();

	HSceneData* GetSceneData() { return &SceneData; }

private:
	HSceneData SceneData;
};

#endif // SCENE_H
