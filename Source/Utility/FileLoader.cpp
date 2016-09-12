#include <Utility/FileLoader.h>

bool HFileLoader::LoadOBJ(const char* filePath,
						  HTriangleMesh &mesh) {

	printf("Loading OBJ file %s.\n", filePath);

	std::vector<unsigned int> vertexIDs;
	std::vector<unsigned int> UVIDs;
	std::vector<unsigned int> normalIDs;
	std::vector<glm::vec3> tempVertices;
	std::vector<glm::vec2> tempUVs;
	std::vector<glm::vec3> tempNormals;

	FILE *file = fopen(filePath, "r");
	if (!file) {
		printf("Could not load file %s.\n", filePath);
		return false;
	}

	while (1) {

		char lineTag[128];
		int res = fscanf(file, "%s", lineTag);
		if (res == EOF) {
			break;
		}

		if (strcmp(lineTag, "v") == 0) {
			glm::vec3 vertex;
			fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
			tempVertices.push_back(vertex);
		}
		else if (strcmp(lineTag, "vt") == 0) {
			glm::vec2 UV;
			fscanf(file, "%f %f\n", &UV.x, &UV.y);
			UV.y = -UV.y;
			tempUVs.push_back(UV);
		}
		else if (strcmp(lineTag, "vn") == 0) {
			glm::vec3 normal;
			fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z);
			tempNormals.push_back(normal);
		}
		else if (strcmp(lineTag, "f") == 0) {
			unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
			/*int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n",
								 &vertexIndex[0], &uvIndex[0], &normalIndex[0],
								 &vertexIndex[1], &uvIndex[1], &normalIndex[1],
								 &vertexIndex[2], &uvIndex[2], &normalIndex[2]);
								 if (matches != 9)
								 {
								 printf("Invalid line formatting.\n");
								 return false;
								 }*/

								 //mesh.triangles.emplace_back(tempVertices[vertexIndex[0] - 1],
								 //tempVertices[vertexIndex[1] - 1],
								 //tempVertices[vertexIndex[2] - 1]);


								 //vertexIDs.push_back(vertexIndex[0]);
								 //vertexIDs.push_back(vertexIndex[1]);
								 //vertexIDs.push_back(vertexIndex[2]);
								 //UVIDs.push_back(uvIndex[0]);
								 //UVIDs.push_back(uvIndex[1]);
								 //UVIDs.push_back(uvIndex[2]);
								 //normalIDs.push_back(normalIndex[0]);
								 //normalIDs.push_back(normalIndex[1]);
								 //normalIDs.push_back(normalIndex[2]);

			int matches = fscanf(file, "%d %d %d\n",
								 &vertexIndex[0],
								 &vertexIndex[1],
								 &vertexIndex[2]);
			if (matches != 3) {
				printf("Invalid line formatting.\n");
				return false;
			}

			mesh.triangles.emplace_back(5.0f*tempVertices[vertexIndex[0] - 1],
										5.0f*tempVertices[vertexIndex[1] - 1],
										5.0f*tempVertices[vertexIndex[2] - 1]);

		}
		else {
			char stupidBuffer[1000];
			fgets(stupidBuffer, 1000, file);
		}

	}

	/*for (unsigned int i = 0; i < vertexIDs.size(); i++)
	{

	unsigned int vertexID = vertexIDs[i];
	unsigned int UVID = UVIDs[i];
	unsigned int normalID = normalIDs[i];

	glm::vec3 vertex = tempVertices[vertexID - 1];
	glm::vec2 UV = tempUVs[UVID - 1];
	glm::vec3 normal = tempNormals[normalID - 1];

	mesh.vertices.push_back(vertex);
	mesh.UVs.push_back(UV);
	mesh.normals.push_back(normal);

	}*/

	return true;

}
