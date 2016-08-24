#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <Core/Include.h>
#include <Core/Shape.h>
#include <Core/Geometry.h>
#include <Core/Material.h>
#include <Utility/MathUtility.h>

struct HTriangle;

struct HTriangleMesh
{

	HTriangleMesh() {}

	~HTriangleMesh() {}

	std::vector<glm::vec3> vertices;
	std::vector<glm::vec2> UVs;
	std::vector<glm::vec3> normals;

	std::vector<HTriangle> triangles;

	static std::vector<HTriangleMesh> meshes;

};

struct HTriangle : HShape
{

	HTriangle() {}

	HTriangle(glm::vec3 vert0, glm::vec3 vert1, glm::vec3 vert2)
		: v0(vert0), v1(vert1), v2(vert2) {}

	__host__ __device__ bool Intersect(HRay &ray, float &t,
									   HSurfaceInteraction &intersection)
	{

		glm::vec3 e1 = v1 - v0;
		glm::vec3 e2 = v2 - v0;
		glm::vec3 P = cross(ray.direction, e2);
		float det = dot(e1, P);

		// Not culling back-facing triangles
		if (det > -M_EPSILON && det < M_EPSILON)
		{
			return false;
		}

		float invDet = 1.0f / det;
		glm::vec3 T = ray.origin - v0;
		float u = dot(T, P)*invDet;

		if (u < 0.0f || u > 1.0f)
		{
			return false;
		}

		glm::vec3 Q = cross(T, e1);
		float v = dot(ray.direction, Q)*invDet;

		if (v < 0.0f || u + v > 1.0f)
		{
			return false;
		}

		float t0 = dot(e2, Q) * invDet;

		if (t0 > M_EPSILON && t0 < t)
		{
			t = t0;
			intersection.position = ray.origin + t*ray.direction;
			intersection.normal = normalize(cross(e1, e2));

			return true;
		}

		return false;

	}


	__host__ __device__ HBoundingBox Bounds() const { return Union(HBoundingBox(v0, v1), v2); }
	//const int meshID;

	glm::vec3 v0;
	glm::vec3 v1;
	glm::vec3 v2;

};


#endif // TRIANGLE_H