#include "Scene.h"

HScene::HScene()
{

}

HScene::~HScene() {}

void HScene::LoadSceneFile()
{
	// Temporary scene setup, should read from file
	NumSpheres = 3;
	Spheres = new HSphere[NumSpheres];

	// Yellow sphere
	Spheres[0].Position = make_float3(-0.9f, 0.0f, -0.3f);
	Spheres[0].Radius = 0.8f;
	Spheres[0].Material.Diffuse = make_float3(0.87f, 0.87f, 0.15f);
	Spheres[0].Material.Emissive = make_float3(0.0f, 0.0f, 0.0f);

	// Blue sphere
	Spheres[1].Position = make_float3(0.8f, 0.0f, -0.8f);
	Spheres[1].Radius = 0.8f;
	Spheres[1].Material.Diffuse = make_float3(0.15f, 0.15f, 0.87f);
	Spheres[1].Material.Emissive = make_float3(0.0f, 0.0f, 0.0f);

	// Light sphere
	Spheres[2].Position = make_float3(1.3f, 1.6f, -2.3f);
	Spheres[2].Radius = 0.8f;
	Spheres[2].Material.Diffuse = make_float3(0.0f, 0.0f, 0.0f);
	Spheres[2].Material.Emissive = make_float3(5.0f, 5.0f, 5.4f);
}
