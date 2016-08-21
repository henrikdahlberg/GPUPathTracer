#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cuda_runtime.h>
#include <math.h>

#include "Material.h"
#include "Utility/MathUtility.h"

// TODO: When I have a scene manager implemented, make it have a ComputeNormals-function
//		 and use precomputed normals in the intersection tests

struct HRay
{

	__host__ __device__ HRay()
		: mint(0.0f), maxt(M_INF)
	{

	}

	__host__ __device__ HRay(const float3 &o,
							 const float3 &d,
							 float start,
							 float end = M_INF)
		: origin(o), direction(d), mint(start), maxt(end)
	{

	}

	__host__ __device__ float3 operator()(float t) const
	{

		return origin + t*direction;

	}

	float3 origin;
	float3 direction;
	float mint;
	float maxt;

};

struct HIntersection
{

	__host__ __device__ HIntersection()
	{

	}

	__host__ __device__ HIntersection(const float3 &p, const float3 &n)
		: position(p), normal(n)
	{

	}

	float3 position;
	float3 normal;

};

struct HSphere
{

	__host__ __device__ bool Intersect(HRay &ray, float &t, HIntersection &intersection);

	float3 position;
	float radius;

	HMaterial material;

};

struct HTriangle
{

	__host__ __device__ bool Intersect(HRay &ray, float &t, HIntersection &intersection);

	float3 v0;
	float3 v1;
	float3 v2;

	HMaterial material;

};

struct HBoundingBox
{

	__host__ __device__ HBoundingBox()
	{

		pmin = make_float3(M_INF, M_INF, M_INF);
		pmax = make_float3(-M_INF, -M_INF, -M_INF);

	}

	__host__ __device__ HBoundingBox(const float3 &p)
		: pmin(p), pmax(p)
	{

	}

	__host__ __device__ HBoundingBox(const float3 &p1, const float3 &p2)
		: pmin(p1), pmax(p2)
	{

	}

	__host__ __device__ const float3 &operator[](int i) const;

	__host__ __device__ float3 &operator[](int i);

	__host__ __device__ bool operator==(const HBoundingBox &b) const
	{
		
		return ((b.pmin.x == pmin.x && b.pmax.x == pmax.x) &&
				(b.pmin.y == pmin.y && b.pmax.y == pmax.y) &&
				(b.pmin.z == pmin.z && b.pmax.z == pmax.z));

	}

	__host__ __device__ bool operator!=(const HBoundingBox &b) const
	{

		return ((b.pmin.x != pmin.x || b.pmax.x != pmax.x) ||
				(b.pmin.y != pmin.y || b.pmax.y != pmax.y) ||
				(b.pmin.z != pmin.z || b.pmax.z != pmax.z));

	}

	__host__ __device__ float3 Diagonal() const
	{

		return pmax - pmin;

	}

	__host__ __device__ float SurfaceArea() const
	{

		float3 d = Diagonal();
		return 2 * (d.x * d.y + d.y * d.z + d.z * d.x);

	}

	__host__ __device__ float Volume() const
	{

		float3 d = Diagonal();
		return d.x * d.y * d.z;

	}

	__host__ __device__ int MaximumDimension() const
	{

		float3 d = Diagonal();
		return (d.x > d.y && d.x > d.z) ? 0 : ((d.y > d.z) ? 1 : 2);

	}

	__host__ __device__ float3 Offset(const float3 &p) const
	{

		float3 o = p - pmin;
		if (pmax.x > pmin.x) o.x /= pmax.x - pmin.x;
		if (pmax.y > pmin.y) o.y /= pmax.y - pmin.y;
		if (pmax.z > pmin.z) o.z /= pmax.z - pmin.z;
		return o;

	}

	__host__ __device__ bool Intersect(const HRay &ray, float* t0, float* t1) const;

	float3 pmin;
	float3 pmax;

};

//////////////////////////////////////////////////////////////////////////
// Geometry inline functions
//////////////////////////////////////////////////////////////////////////

__host__ __device__ inline bool HSphere::Intersect(HRay &ray, float &t, HIntersection &intersection)
{

	float3 op = position - ray.origin;
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

__host__ __device__ inline bool HTriangle::Intersect(HRay &ray, float &t, HIntersection &intersection)
{

	float3 e1 = v1 - v0;
	float3 e2 = v2 - v0;
	float3 P = cross(ray.direction, e2);
	float det = dot(e1, P);

	// Not culling back-facing triangles
	if (det > -M_EPSILON && det < M_EPSILON)
	{
		return false;
	}

	float invDet = 1.0f / det;
	float3 T = ray.origin - v0;
	float u = dot(T, P)*invDet;

	if (u < 0.0f || u > 1.0f)
	{
		return false;
	}

	float3 Q = cross(T, e1);
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
		intersection.normal = normalize(cross(e1,e2));

		return true;
	}

	return false;

}

__host__ __device__ inline HBoundingBox Union(const HBoundingBox &b, const float3 &p)
{

	return HBoundingBox(make_float3(fmin(b.pmin.x, p.x),
									fmin(b.pmin.y, p.y),
									fmin(b.pmin.z, p.z)),
						make_float3(fmax(b.pmax.x, p.x),
									fmax(b.pmax.y, p.y),
									fmax(b.pmax.z, p.z)));

}

__host__ __device__ inline HBoundingBox Union(const HBoundingBox &b1, const HBoundingBox &b2)
{

	return HBoundingBox(make_float3(fmin(b1.pmin.x, b2.pmin.x),
									fmin(b1.pmin.y, b2.pmin.y),
									fmin(b1.pmin.z, b2.pmin.z)),
						make_float3(fmax(b1.pmax.x, b2.pmax.x),
									fmax(b1.pmax.y, b2.pmax.y),
									fmax(b1.pmax.z, b2.pmax.z)));

}

__host__ __device__ inline HBoundingBox Intersection(const HBoundingBox &b1, const HBoundingBox &b2)
{

	return HBoundingBox(make_float3(fmax(b1.pmin.x, b2.pmin.x),
									fmax(b1.pmin.y, b2.pmin.y),
									fmax(b1.pmin.z, b2.pmin.z)),
						make_float3(fmin(b1.pmax.x, b2.pmax.x),
									fmin(b1.pmax.y, b2.pmax.y),
									fmin(b1.pmax.z, b2.pmax.z)));

}

__host__ __device__ inline bool Overlaps(const HBoundingBox &b1, const HBoundingBox &b2)
{

	bool x = (b1.pmax.x >= b2.pmin.x) && (b1.pmin.x <= b2.pmax.x);
	bool y = (b1.pmax.y >= b2.pmin.y) && (b1.pmin.y <= b2.pmax.y);
	bool z = (b1.pmax.z >= b2.pmin.z) && (b1.pmin.z <= b2.pmax.z);
	return (x && y && z);

}

__host__ __device__ inline bool Contains(const HBoundingBox &b, const float3 &p)
{

	return (p.x >= b.pmin.x && p.x <= b.pmax.x &&
			p.y >= b.pmin.y && p.y <= b.pmax.y &&
			p.z >= b.pmin.z && p.z <= b.pmax.z);

}

__host__ __device__ inline void BoundingSphere(const HBoundingBox &b, float3* position, float* radius)
{

	*position = (b.pmin + b.pmax) * 0.5f;
	*radius = Contains(b, *position) ? length(*position - b.pmax) : 0;

}

__host__ __device__ inline bool ContainsExclusive(const HBoundingBox &b, const float3 &p)
{
	return (p.x >= b.pmin.x && p.x < b.pmax.x &&
			p.y >= b.pmin.y && p.y < b.pmax.y &&
			p.z >= b.pmin.z && p.z < b.pmax.z);
}

__host__ __device__ inline HBoundingBox Expand(const HBoundingBox &b, const float delta)
{

	return HBoundingBox(b.pmin - make_float3(delta),
						b.pmax + make_float3(delta));

}

__host__ __device__ inline bool HBoundingBox::Intersect(const HRay &ray, float* t0, float* t1) const
{

	// TODO
	return false;

}


#endif // GEOMETRY_H
