#include <Core/Include.h>

#include <Core/Scene.h>
#include <Shapes/Sphere.h>

void HScene::LoadSceneFile() {
	HMaterial white = HMaterial(glm::vec3(0.95f), glm::vec3(0.0f));
	white.materialType = DIFFUSE;
	HMaterial red = HMaterial(glm::vec3(0.7f, 0.01f, 0.01f), glm::vec3(0.0f));
	red.materialType = DIFFUSE;
	HMaterial green = HMaterial(glm::vec3(0.01f, 0.3f, 0.01f), glm::vec3(0.0f));
	green.materialType = DIFFUSE;
	HMaterial cornellLight = HMaterial(glm::vec3(0.0f), 20.0f * glm::vec3(0.736507f, 0.642866f, 0.210431f));
	cornellLight.materialType = DIFFUSE;

	HMaterial brightLight = HMaterial(glm::vec3(0.0f), glm::vec3(20.0f, 20.0f, 17.0f));
	brightLight.materialType = DIFFUSE;

	HMaterial glossyRed = HMaterial(glm::vec3(0.87f, 0.15f, 0.15f), glm::vec3(0.0f), glm::vec3(1.0f), 1.491f);
	glossyRed.materialType = SPECULAR;

	HMaterial glossyGreen = HMaterial(glm::vec3(0.15f, 0.87f, 0.15f), glm::vec3(0.0f), glm::vec3(1.0f), 1.491f);
	glossyGreen.materialType = SPECULAR;

	HMaterial glossyLtBlue = HMaterial(glm::vec3(0.4f, 0.6f, 0.8f), glm::vec3(0.0f), glm::vec3(1.0f), 1.491f);
	glossyLtBlue.materialType = SPECULAR;

	HMaterial glossyOrange = HMaterial(glm::vec3(0.93f, 0.33f, 0.04f), glm::vec3(0.0f), glm::vec3(1.0f), 1.491f);
	glossyOrange.materialType = SPECULAR;

	HMaterial glossyPurple = HMaterial(glm::vec3(0.5f, 0.1f, 0.9f), glm::vec3(0.0f), glm::vec3(1.0f), 1.491f);
	glossyPurple.materialType = SPECULAR;

	HMaterial glass = HMaterial(glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(1.0f), 1.62f);
	glass.materialType = ALL_TRANS;

	HMaterial greenGlass = HMaterial(glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(1.0f), 1.62f);
	greenGlass.medium.scatteringProperties = HScatteringProperties(0.0f, glm::vec3(1.0f, 0.01f, 1.0f));
	greenGlass.materialType = ALL_TRANS;

	HMaterial mirror = HMaterial(glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(1.0f), 1.9f);
	mirror.materialType = REFLECTION;

	HMaterial blueSub = HMaterial(glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(1.0f), 1.1f);
	blueSub.medium.scatteringProperties = HScatteringProperties(16.0f, glm::vec3(10.0f, 3.0f, 0.02f));
	blueSub.materialType = ALL_TRANS;

	HMaterial greenSub = HMaterial(glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(1.0f), 1.4f);
	greenSub.medium.scatteringProperties = HScatteringProperties(16.0f, glm::vec3(10.0f, 0.2f, 6.0f));
	greenSub.materialType = ALL_TRANS;

	HMaterial redSub = HMaterial(glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(1.0f), 1.35f);
	redSub.medium.scatteringProperties = HScatteringProperties(9.0f, glm::vec3(0.02f, 5.1f, 5.7f));
	redSub.materialType = ALL_TRANS;

	HMaterial yellowSub = HMaterial(glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(1.0f), 1.35f);
	yellowSub.medium.scatteringProperties = HScatteringProperties(9.0f, glm::vec3(0.02f, 0.57f, 5.7f));
	yellowSub.materialType = ALL_TRANS;

	HMaterial orangeSub = HMaterial(glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(1.0f), 1.35f);
	orangeSub.medium.scatteringProperties = HScatteringProperties(32.0f, glm::vec3(0.02f, 8.0f, 50.f));
	orangeSub.materialType = ALL_TRANS;

	HMaterial marble = HMaterial(glm::vec3(0.0f), glm::vec3(0.0f), glm::vec3(1.0f), 1.486f);
	marble.medium.scatteringProperties = HScatteringProperties(64.0f, glm::vec3(0.0f));
	marble.materialType = ALL_TRANS;


	triangles.clear();

	//////////////////////////////////////////////////////////////////////////
	// Random colored sphere scene
	//////////////////////////////////////////////////////////////////////////
	//numSpheres = 300;
	//spheres = new HSphere[numSpheres];

	//srand(time(nullptr));

	//for (int i = 0; i < numSpheres - 1; i++) {

	//	spheres[i].position = glm::vec3(2.0f - 4.0f*(float)rand() / (float)RAND_MAX,
	//									0.1f + 1.4f*(float)rand() / (float)RAND_MAX,
	//									-4.0f*(float)rand() / (float)RAND_MAX);
	//	spheres[i].radius = 0.01f + 0.2f*(float)rand() / (float)RAND_MAX;
	//	spheres[i].material.diffuse = glm::vec3((float)rand() / (float)RAND_MAX,
	//											(float)rand() / (float)RAND_MAX,
	//											(float)rand() / (float)RAND_MAX);

	//	if ((float)rand() / (float)RAND_MAX > 0.8f) {
	//		spheres[i].material.emission = glm::vec3(5.0f * (float)rand() / (float)RAND_MAX,
	//												 5.0f * (float)rand() / (float)RAND_MAX,
	//												 5.4f * (float)rand() / (float)RAND_MAX);
	//	}
	//	else {
	//		spheres[i].material.emission = glm::vec3(0.0f, 0.0f, 0.0f);
	//	}
	//	spheres[i].material.materialType = DIFFUSE;
	//}

	//spheres[numSpheres - 1].position = glm::vec3(0.0f, 3.0f, 0.0f);
	//spheres[numSpheres - 1].radius = 0.8f;
	//spheres[numSpheres - 1].material.diffuse = glm::vec3(0.0f, 0.0f, 0.0f);
	//spheres[numSpheres - 1].material.emission = glm::vec3(5.0f, 5.0f, 5.4f);
	//spheres[numSpheres - 1].material.materialType = DIFFUSE;

	//////////////////////////////////////////////////////////////////////////
	// Fresnel test scene
	//////////////////////////////////////////////////////////////////////////
	//numSpheres = 11;
	numSpheres = 1;
	spheres = new HSphere[numSpheres];

	spheres[0].position = glm::vec3(0.4f, 10.0f, -1.0f);
	spheres[0].radius = 1.5f;
	spheres[0].material = brightLight;

	//spheres[0].position = glm::vec3(-0.4f, 0.05f, 0.1f);
	//spheres[0].radius = 0.15f;
	//spheres[0].material = mirror;

	//spheres[1].position = glm::vec3(0.9f, 0.03f, 0.2f);
	//spheres[1].radius = 0.13f;
	//spheres[1].material = greenGlass;

	//spheres[2].position = glm::vec3(0.9f, 0.09f, -0.15f);
	//spheres[2].radius = 0.19f;
	////spheres[2].material = glossyPurple;
	//spheres[2].material = glossyLtBlue;

	//spheres[3].position = glm::vec3(1.0f, -0.06f, 0.4f);
	//spheres[3].radius = 0.04f;
	////spheres[3].material = brightLight;
	//spheres[3].material = mirror;

	//spheres[4].position = glm::vec3(1.5f, 0.3f, 0.0f);
	//spheres[4].radius = 0.4f;
	//spheres[4].material = mirror;

	//spheres[5].position = glm::vec3(0.65f, 0.02f, 0.2f);
	//spheres[5].radius = 0.12f;
	//spheres[5].material = glossyPurple;

	//spheres[6].position = glm::vec3(0.9f, 0.03f + 0.38f, -0.15f);
	//spheres[6].radius = 0.13f;
	////spheres[6].material = glossyLtBlue;
	//spheres[6].material = greenSub;

	//spheres[7].position = glm::vec3(1.52f, 2.2f/*1.59f*/, -3.12f);
	//spheres[7].radius = 0.6f;
	//spheres[7].material = brightLight;

	//spheres[8].position = glm::vec3(0.9f, 0.26f, 0.2f);
	//spheres[8].radius = 0.10f;
	////spheres[8].material = mirror;
	//spheres[8].material = redSub;

	//spheres[9].position = glm::vec3(1.5f, 0.1f, 0.7f);
	//spheres[9].radius = 0.2f;
	////spheres[9].material = glossyGreen;
	//spheres[9].material = orangeSub;

	//spheres[10].position = glm::vec3(1.5f, 0.46f, 0.7f);
	//spheres[10].radius = 0.16f;
	////spheres[10].material = glossyRed;
	//spheres[10].material = yellowSub;

	//HMaterial floorMaterial = HMaterial(glm::vec3(0.1f), glm::vec3(0.0f));
	//floorMaterial.materialType = DIFFUSE;
	//numTriangles = 2;
	//triangles = new HTriangle[numTriangles];
	//triangles[0] = HTriangle(glm::vec3(-25.0f, -0.1f, -25.0f),
	//						 glm::vec3(25.0f, -0.1f, -25.0f),
	//						 glm::vec3(25.0f, -0.1f, 25.0f),
	//						 floorMaterial);
	//triangles[1] = HTriangle(glm::vec3(-25.0f, -0.1f, -25.0f),
	//						 glm::vec3(25.0f, -0.1f, 25.0f),
	//						 glm::vec3(-25.0f, -0.1f, 25.0f),
	//						 floorMaterial);


	////////////////////////////////////////////////////////////////////////////
	//// Cornell box
	////////////////////////////////////////////////////////////////////////////

	//glm::vec3 A, B, C, D, E, F, G, H, I, J, K, L;

	//// Bottom vertices
	//A = glm::vec3(-552.8f, 0.0f, 0.0f);
	//B = glm::vec3(0.0f, 0.0f, 0.0f);
	//C = glm::vec3(0.0f, 0.0f, -559.2f);
	//D = glm::vec3(-549.6f, 0.0f, -559.2f);

	//// Top vertices
	//E = glm::vec3(-556.0f, 548.8f, 0.0f);
	//F = glm::vec3(-556.0f, 548.8f, -559.2f);
	//G = glm::vec3(0.0f, 548.8f, -559.2f);
	//H = glm::vec3(0.0f, 548.8f, 0.0f);

	//// Light/roof hole vertices
	//I = glm::vec3(-343.0f, 548.8f, -227.0f);
	//J = glm::vec3(-343.0f, 548.8f, -332.0f);
	//K = glm::vec3(-213.0f, 548.8f, -332.0f);
	//L = glm::vec3(-213.0f, 548.8f, -227.0f);

	////------------------------------------------------------------------------
	//// Floor
	////------------------------------------------------------------------------
	//triangles.push_back(HTriangle(A, B, C, white));
	//triangles.push_back(HTriangle(A, C, D, white));
	////------------------------------------------------------------------------

	////------------------------------------------------------------------------
	//// Roof
	////------------------------------------------------------------------------
	//triangles.push_back(HTriangle(E, F, I, white));
	//triangles.push_back(HTriangle(F, G, J, white));
	//triangles.push_back(HTriangle(G, H, K, white));
	//triangles.push_back(HTriangle(H, E, L, white));
	//triangles.push_back(HTriangle(F, J, I, white));
	//triangles.push_back(HTriangle(G, K, J, white));
	//triangles.push_back(HTriangle(H, L, K, white));
	//triangles.push_back(HTriangle(E, I, L, white));

	//// Light
	//triangles.push_back(HTriangle(I, J, K, cornellLight));
	//triangles.push_back(HTriangle(I, K, L, cornellLight));
	////------------------------------------------------------------------------

	////------------------------------------------------------------------------
	//// Back wall
	////------------------------------------------------------------------------
	//triangles.push_back(HTriangle(D, C, G, white));
	//triangles.push_back(HTriangle(D, G, F, white));
	////------------------------------------------------------------------------

	////------------------------------------------------------------------------
	//// Right wall
	////------------------------------------------------------------------------
	//triangles.push_back(HTriangle(B, H, G, green));
	//triangles.push_back(HTriangle(B, G, C, green));
	////------------------------------------------------------------------------

	////------------------------------------------------------------------------
	//// Left wall
	////------------------------------------------------------------------------
	//triangles.push_back(HTriangle(A, D, F, red));
	//triangles.push_back(HTriangle(A, F, E, red));
	////------------------------------------------------------------------------

	////------------------------------------------------------------------------
	//// Short block
	////------------------------------------------------------------------------
	//glm::vec3 sA, sB, sC, sD, sE, sF, sG, sH;

	//sA = glm::vec3(-290.0f, 0.0f, -114.0f);
	//sB = glm::vec3(-240.0f, 0.0f, -272.0f);
	//sC = glm::vec3(-82.0f, 0.0f, -225.0f);
	//sD = glm::vec3(-130.0f, 0.0f, -65.0f);
	//sE = glm::vec3(-290.0f, 165.0f, -114.0f);
	//sF = glm::vec3(-240.0f, 165.0f, -272.0f);
	//sG = glm::vec3(-82.0f, 165.0f, -225.0f);
	//sH = glm::vec3(-130.0f, 165.0f, -65.0f);

	//triangles.push_back(HTriangle(sA, sE, sF, white));
	//triangles.push_back(HTriangle(sA, sF, sB, white));
	//triangles.push_back(HTriangle(sB, sF, sG, white));
	//triangles.push_back(HTriangle(sB, sG, sC, white));
	//triangles.push_back(HTriangle(sC, sG, sH, white));
	//triangles.push_back(HTriangle(sC, sH, sD, white));
	//triangles.push_back(HTriangle(sD, sH, sE, white));
	//triangles.push_back(HTriangle(sD, sE, sA, white));
	//triangles.push_back(HTriangle(sA, sE, sF, white));
	//triangles.push_back(HTriangle(sA, sF, sB, white));
	//triangles.push_back(HTriangle(sE, sF, sG, white));
	//triangles.push_back(HTriangle(sE, sG, sH, white));
	////------------------------------------------------------------------------

	////------------------------------------------------------------------------
	//// Tall block
	////------------------------------------------------------------------------
	//glm::vec3 tA, tB, tC, tD, tE, tF, tG, tH;

	//tA = glm::vec3(-423.0f, 0.0f, -247.0f);
	//tB = glm::vec3(-472.0f, 0.0f, -406.0f);
	//tC = glm::vec3(-314.0f, 0.0f, -456.0f);
	//tD = glm::vec3(-265.0f, 0.0f, -296.0f);
	//tE = glm::vec3(-423.0f, 330.0f, -247.0f);
	//tF = glm::vec3(-472.0f, 330.0f, -406.0f);
	//tG = glm::vec3(-314.0f, 330.0f, -456.0f);
	//tH = glm::vec3(-265.0f, 330.0f, -296.0f);

	//triangles.push_back(HTriangle(tA, tE, tF, white));
	//triangles.push_back(HTriangle(tA, tF, tB, white));
	//triangles.push_back(HTriangle(tB, tF, tG, white));
	//triangles.push_back(HTriangle(tB, tG, tC, white));
	//triangles.push_back(HTriangle(tC, tG, tH, white));
	//triangles.push_back(HTriangle(tC, tH, tD, white));
	//triangles.push_back(HTriangle(tD, tH, tE, white));
	//triangles.push_back(HTriangle(tD, tE, tA, white));
	//triangles.push_back(HTriangle(tE, tF, tG, white));
	//triangles.push_back(HTriangle(tE, tG, tH, white));
	////------------------------------------------------------------------------

	//// Scale measured data down to unit box
	//float scale = 1.0f / 559.2f;
	//glm::vec3 offset = glm::vec3(0.5f, 0.0f, 0.5f);
	//for (HTriangle& t : triangles) {
	//	t.v0 *= scale;
	//	t.v1 *= scale;
	//	t.v2 *= scale;
	//	t.v0 += offset;
	//	t.v1 += offset;
	//	t.v2 += offset;
	//}

	//numSpheres = 2;
	//spheres = new HSphere[numSpheres];

	//spheres[0].radius = 0.18f;
	//spheres[0].position = glm::vec3(0.25f, 0.18f + 0.2951f, 0.15f);
	//spheres[0].material = glossyLtBlue;

	//spheres[1].radius = 0.13f;
	//spheres[1].position = glm::vec3(-0.30f, 0.13f, 0.3f);
	//spheres[1].material = glass;
	////------------------------------------------------------------------------

	// Temp meshloading
	HTriangleMesh mesh;
	fileLoader.LoadOBJ("dragon.obj", mesh);
	mesh.position = glm::vec3(0.0f, -0.4f, 0.0f);
	mesh.scale = 0.25f;
	for (HTriangle tri : mesh.triangles) {
		tri.v0 += mesh.position;
		tri.v0 *= mesh.scale;
		tri.v1 += mesh.position;
		tri.v1 *= mesh.scale;
		tri.v2 += mesh.position;
		tri.v2 *= mesh.scale;
		tri.material = greenSub;
		//tri.material = red;
		triangles.push_back(tri);
	}

	HMaterial floorMaterial = HMaterial(glm::vec3(0.2f), glm::vec3(0.0f));
	floorMaterial.materialType = DIFFUSE;
	triangles.push_back(HTriangle(glm::vec3(-25.0f, -0.1f, -25.0f),
		glm::vec3(25.0f, -0.1f, -25.0f),
		glm::vec3(25.0f, -0.1f, 25.0f),
		glossyLtBlue));
	triangles.push_back(HTriangle(glm::vec3(-25.0f, -0.1f, -25.0f),
		glm::vec3(25.0f, -0.1f, 25.0f),
		glm::vec3(-25.0f, -0.1f, 25.0f),
		glossyLtBlue));


	

	numTriangles = triangles.size();


}
