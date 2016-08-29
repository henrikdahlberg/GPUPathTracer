#include <Core/Include.h>

#include <Core/Scene.h>
#include <Shapes/Sphere.h>

void HScene::LoadSceneFile() {
	// Random colored sphere scene
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
	//}

	////spheres[numSpheres - 2].position = glm::vec3(0.0f, -1e5f, 0.0f);
	////spheres[numSpheres - 2].radius = 1e5f;
	////spheres[numSpheres - 2].material.diffuse = glm::vec3(0.15f, 0.15f, 0.15f);
	////spheres[numSpheres - 2].material.emission = glm::vec3(0.0f, 0.0f, 0.0f);

	//spheres[numSpheres - 1].position = glm::vec3(0.0f, 3.0f, 0.0f);
	//spheres[numSpheres - 1].radius = 0.8f;
	//spheres[numSpheres - 1].material.diffuse = glm::vec3(0.0f, 0.0f, 0.0f);
	//spheres[numSpheres - 1].material.emission = glm::vec3(5.0f, 5.0f, 5.4f);

	/*HMaterial floorMaterial = HMaterial(glm::vec3(0.4f), glm::vec3(0.0f));
	numTriangles = 2;
	triangles = new HTriangle[numTriangles];
	triangles[0] = HTriangle(glm::vec3(-25.0f, -0.1f, -25.0f),
	glm::vec3(25.0f, -0.1f, -25.0f),
	glm::vec3(25.0f, -0.1f, 25.0f),
	floorMaterial);
	triangles[1] = HTriangle(glm::vec3(-25.0f, -0.1f, -25.0f),
	glm::vec3(25.0f, -0.1f, 25.0f),
	glm::vec3(-25.0f, -0.1f, 25.0f),
	floorMaterial);*/

	//////////////////////////////////////////////////////////////////////////
	// Cornell box
	//////////////////////////////////////////////////////////////////////////
	HMaterial white = HMaterial(glm::vec3(0.95f), glm::vec3(0.0f));
	HMaterial red = HMaterial(glm::vec3(0.7f, 0.01f, 0.01f), glm::vec3(0.0f));
	HMaterial green = HMaterial(glm::vec3(0.01f, 0.3f, 0.01f), glm::vec3(0.0f));
	HMaterial light = HMaterial(glm::vec3(0.0f), 20.0f * glm::vec3(0.736507f, 0.642866f, 0.210431f));

	numTriangles = 40;

	triangles = new HTriangle[numTriangles];
	glm::vec3 A, B, C, D, E, F, G, H, I, J, K, L;

	// Bottom vertices
	A = glm::vec3(-552.8f, 0.0f, 0.0f);
	B = glm::vec3(0.0f, 0.0f, 0.0f);
	C = glm::vec3(0.0f, 0.0f, -559.2f);
	D = glm::vec3(-549.6f, 0.0f, -559.2f);

	// Top vertices
	E = glm::vec3(-556.0f, 548.8f, 0.0f);
	F = glm::vec3(-556.0f, 548.8f, -559.2f);
	G = glm::vec3(0.0f, 548.8f, -559.2f);
	H = glm::vec3(0.0f, 548.8f, 0.0f);

	// Light/roof hole vertices
	I = glm::vec3(-343.0f, 548.8f, -227.0f);
	J = glm::vec3(-343.0f, 548.8f, -332.0f);
	K = glm::vec3(-213.0f, 548.8f, -332.0f);
	L = glm::vec3(-213.0f, 548.8f, -227.0f);

	//------------------------------------------------------------------------
	// Floor
	//------------------------------------------------------------------------
	triangles[0] = HTriangle(A, B, C, white);
	triangles[1] = HTriangle(A, C, D, white);
	//------------------------------------------------------------------------

	//------------------------------------------------------------------------
	// Roof
	//------------------------------------------------------------------------
	triangles[2] = HTriangle(E, F, I, white);
	triangles[3] = HTriangle(F, G, J, white);
	triangles[4] = HTriangle(G, H, K, white);
	triangles[5] = HTriangle(H, E, L, white);
	triangles[6] = HTriangle(F, J, I, white);
	triangles[7] = HTriangle(G, K, J, white);
	triangles[8] = HTriangle(H, L, K, white);
	triangles[9] = HTriangle(E, I, L, white);

	// Light
	triangles[10] = HTriangle(I, J, K, light);
	triangles[11] = HTriangle(I, K, L, light);
	//------------------------------------------------------------------------

	//------------------------------------------------------------------------
	// Back wall
	//------------------------------------------------------------------------
	triangles[12] = HTriangle(D, C, G, white);
	triangles[13] = HTriangle(D, G, F, white);
	//------------------------------------------------------------------------

	//------------------------------------------------------------------------
	// Right wall
	//------------------------------------------------------------------------
	triangles[14] = HTriangle(B, H, G, green);
	triangles[15] = HTriangle(B, G, C, green);
	//------------------------------------------------------------------------

	//------------------------------------------------------------------------
	// Left wall
	//------------------------------------------------------------------------
	triangles[16] = HTriangle(A, D, F, red);
	triangles[17] = HTriangle(A, F, E, red);
	//------------------------------------------------------------------------

	//------------------------------------------------------------------------
	// Short block
	//------------------------------------------------------------------------
	glm::vec3 sA, sB, sC, sD, sE, sF, sG, sH;

	sA = glm::vec3(-290.0f, 0.0f, -114.0f);
	sB = glm::vec3(-240.0f, 0.0f, -272.0f);
	sC = glm::vec3(-82.0f, 0.0f, -225.0f);
	sD = glm::vec3(-130.0f, 0.0f, -65.0f);
	sE = glm::vec3(-290.0f, 165.0f, -114.0f);
	sF = glm::vec3(-240.0f, 165.0f, -272.0f);
	sG = glm::vec3(-82.0f, 165.0f, -225.0f);
	sH = glm::vec3(-130.0f, 165.0f, -65.0f);

	triangles[18] = HTriangle(sA, sE, sF, white);
	triangles[19] = HTriangle(sA, sF, sB, white);
	triangles[20] = HTriangle(sB, sF, sG, white);
	triangles[21] = HTriangle(sB, sG, sC, white);
	triangles[22] = HTriangle(sC, sG, sH, white);
	triangles[23] = HTriangle(sC, sH, sD, white);
	triangles[24] = HTriangle(sD, sH, sE, white);
	triangles[25] = HTriangle(sD, sE, sA, white);
	triangles[26] = HTriangle(sA, sE, sF, white);
	triangles[27] = HTriangle(sA, sF, sB, white);
	triangles[28] = HTriangle(sE, sF, sG, white);
	triangles[29] = HTriangle(sE, sG, sH, white);
	//------------------------------------------------------------------------

	//------------------------------------------------------------------------
	// Tall block
	//------------------------------------------------------------------------
	glm::vec3 tA, tB, tC, tD, tE, tF, tG, tH;

	tA = glm::vec3(-423.0f, 0.0f, -247.0f);
	tB = glm::vec3(-472.0f, 0.0f, -406.0f);
	tC = glm::vec3(-314.0f, 0.0f, -456.0f);
	tD = glm::vec3(-265.0f, 0.0f, -296.0f);
	tE = glm::vec3(-423.0f, 330.0f, -247.0f);
	tF = glm::vec3(-472.0f, 330.0f, -406.0f);
	tG = glm::vec3(-314.0f, 330.0f, -456.0f);
	tH = glm::vec3(-265.0f, 330.0f, -296.0f);

	triangles[30] = HTriangle(tA, tE, tF, white);
	triangles[31] = HTriangle(tA, tF, tB, white);
	triangles[32] = HTriangle(tB, tF, tG, white);
	triangles[33] = HTriangle(tB, tG, tC, white);
	triangles[34] = HTriangle(tC, tG, tH, white);
	triangles[35] = HTriangle(tC, tH, tD, white);
	triangles[36] = HTriangle(tD, tH, tE, white);
	triangles[37] = HTriangle(tD, tE, tA, white);
	triangles[38] = HTriangle(tE, tF, tG, white);
	triangles[39] = HTriangle(tE, tG, tH, white);
	//------------------------------------------------------------------------

	// Scale measured data down to unit box
	float scale = 1.0f / 559.2f;
	glm::vec3 offset = glm::vec3(0.5f, 0.0f, 0.5f);
	for (int i=0; i < numTriangles; i++) {
		triangles[i].v0 *= scale;
		triangles[i].v1 *= scale;
		triangles[i].v2 *= scale;
		triangles[i].v0 += offset;
		triangles[i].v1 += offset;
		triangles[i].v2 += offset;
	}

	numSpheres = 1;
	spheres = new HSphere[numSpheres];

	spheres[0].radius = 0.18f;
	spheres[0].position = glm::vec3(0.25f, 0.18f + 0.2951f, 0.15f);
	spheres[0].material = white;
	//------------------------------------------------------------------------

	// Temp meshloading
	//HTriangleMesh mesh;
	//fileLoader.LoadOBJ("bunny.obj", mesh);

	//numTriangles = mesh.triangles.size();
	//triangles = new HTriangle[numTriangles];
	//for (int i = 0; i < numTriangles; i++) {
	//	triangles[i] = mesh.triangles[i];
	//	triangles[i].material.diffuse = glm::vec3(0.15f, 0.87f, 0.87f);
	//	triangles[i].material.emission = glm::vec3(0.0f, 0.0f, 0.0f);
	//}

}
