#include <Core/Include.h>

#include <Core/Scene.h>
#include <Shapes/Sphere.h>

void HScene::LoadSceneFile() {
	// Random colored sphere scene
	numSpheres = 300;
	spheres = new HSphere[numSpheres];

	srand(time(nullptr));

	for (int i = 0; i < numSpheres - 1; i++) {

		spheres[i].position = glm::vec3(2.0f - 4.0f*(float)rand() / (float)RAND_MAX,
										0.1f + 1.4f*(float)rand() / (float)RAND_MAX,
										-4.0f*(float)rand() / (float)RAND_MAX);
		spheres[i].radius = 0.01f + 0.2f*(float)rand() / (float)RAND_MAX;
		spheres[i].material.diffuse = glm::vec3((float)rand() / (float)RAND_MAX,
												(float)rand() / (float)RAND_MAX,
												(float)rand() / (float)RAND_MAX);

		if ((float)rand() / (float)RAND_MAX > 0.8f) {
			spheres[i].material.emission = glm::vec3(5.0f * (float)rand() / (float)RAND_MAX,
													 5.0f * (float)rand() / (float)RAND_MAX,
													 5.4f * (float)rand() / (float)RAND_MAX);
		}
		else {
			spheres[i].material.emission = glm::vec3(0.0f, 0.0f, 0.0f);
		}
	}

	//spheres[numSpheres - 2].position = glm::vec3(0.0f, -1e5f, 0.0f);
	//spheres[numSpheres - 2].radius = 1e5f;
	//spheres[numSpheres - 2].material.diffuse = glm::vec3(0.15f, 0.15f, 0.15f);
	//spheres[numSpheres - 2].material.emission = glm::vec3(0.0f, 0.0f, 0.0f);

	spheres[numSpheres - 1].position = glm::vec3(0.0f, 3.0f, 0.0f);
	spheres[numSpheres - 1].radius = 0.8f;
	spheres[numSpheres - 1].material.diffuse = glm::vec3(0.0f, 0.0f, 0.0f);
	spheres[numSpheres - 1].material.emission = glm::vec3(5.0f, 5.0f, 5.4f);

	//////////////////////////////////////////////////////////////////////////
	// Cornell box
	//////////////////////////////////////////////////////////////////////////
	HMaterial white = HMaterial(glm::vec3(0.90f), glm::vec3(0.0f));
	HMaterial red = HMaterial(glm::vec3(0.90f, 0.05f, 0.05f), glm::vec3(0.0f));
	HMaterial green = HMaterial(glm::vec3(0.05f, 0.90f, 0.05f), glm::vec3(0.0f));
	HMaterial light = HMaterial(glm::vec3(0.0f), glm::vec3(20.4f));

	numTriangles = 20;
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

	float scale = 1.0f / 559.2f;
	glm::vec3 offset = glm::vec3(0.5f, 0.0f, 0.5f);
	for (int i=0; i < 18; i++) {
		triangles[i].v0 *= scale;
		triangles[i].v1 *= scale;
		triangles[i].v2 *= scale;
		triangles[i].v0 += offset;
		triangles[i].v1 += offset;
		triangles[i].v2 += offset;
	}
	//------------------------------------------------------------------------

	triangles[18] = HTriangle(glm::vec3(-2.5f, -0.1f, -2.5f),
							  glm::vec3(2.5f, -0.1f, -2.5f),
							  glm::vec3(2.5f, -0.1f, 2.5f), white);
	triangles[19] = HTriangle(glm::vec3(-2.5f, -0.1f, -2.5f),
							  glm::vec3(2.5f, -0.1f, 2.5f),
							  glm::vec3(-2.5f, -0.1f, 2.5f), white);

	// Temp meshloading
	/*HTriangleMesh mesh;
	fileLoader.LoadOBJ("bunny.obj", mesh);

	numTriangles = mesh.triangles.size();
	triangles = new HTriangle[numTriangles];
	for (int i = 0; i < numTriangles; i++)
	{
	triangles[i] = mesh.triangles[i];
	triangles[i].material.diffuse = glm::vec3(0.15f, 0.87f, 0.15f);
	triangles[i].material.emission = glm::vec3(0.0f, 0.0f, 0.0f);
	}*/

}
