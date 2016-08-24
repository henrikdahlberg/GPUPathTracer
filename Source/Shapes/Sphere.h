#ifndef SPHERE_H
#define SPHERE_H


#include <Core/Shape.h>
#include <Core/Geometry.h>
#include <Core/Interaction.h>

struct HSphere : HShape
{

	__host__ __device__ bool Intersect(HRay &ray, float &t,
									   HSurfaceInteraction &intersection)
	{
		glm::vec3 op = position - ray.origin;
		float b = dot(op, ray.direction);
		float discriminant = b*b - dot(op, op) + radius*radius;

		if (discriminant < 0)
		{
			return false;
		}

		discriminant = sqrtf(discriminant);

		float t1 = b - discriminant;
		float t2 = b + discriminant;

		if (t1 > M_EPSILON && t1 < t)
		{
			t = t1;
		}
		else if (t2 > M_EPSILON && t2 < t)
		{
			t = t2;
		}
		else
		{
			return false;
		}

		intersection.position = ray.origin + t*ray.direction;
		intersection.normal = normalize(intersection.position - position);

		return true;
	}

	glm::vec3 position;
	float radius;

};

#endif // SPHERE_H