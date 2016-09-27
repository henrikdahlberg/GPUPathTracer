#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <Core/Include.h>
#include <Core/Material.h>
#include <Utility/MathUtility.h>

// TODO: When I have a scene manager implemented, make it have a ComputeNormals-function
//		 and use precomputed normals in the intersection tests

struct HRay {
	__host__ __device__ HRay()
		: mint(0.0f), maxt(M_INF) {}
	__host__ __device__ HRay(const glm::vec3 &o,
							 const glm::vec3 &d,
							 float start,
							 float end = M_INF)
							 : origin(o), direction(d), mint(start), maxt(end) {}

	__host__ __device__ glm::vec3 operator()(float t) const { return origin + t*direction; }

	glm::vec3 origin;
	glm::vec3 direction;
	float mint;
	float maxt;
	HMedium enteredMedium;
	HMedium currentMedium;
	bool transmitted;
};

struct HBoundingBox {

	__host__ __device__ HBoundingBox() { pmin = glm::vec3(M_INF, M_INF, M_INF);
										 pmax = glm::vec3(-M_INF, -M_INF, -M_INF); }
	__host__ __device__ HBoundingBox(const glm::vec3 &p)
		: pmin(p), pmax(p) {}
	__host__ __device__ HBoundingBox(const glm::vec3 &p1, const glm::vec3 &p2)
		: pmin(p1), pmax(p2) {}

	__host__ __device__ const glm::vec3 &operator[](int i) const { /* TODO: assert i=0, i=1*/ return (i == 0) ? pmin : pmax; }

	__host__ __device__ glm::vec3 &operator[](int i) { /*TODO: assert i=0, i=1*/ return (i == 0) ? pmin : pmax; }

	__host__ __device__ bool operator==(const HBoundingBox &b) const {
		return ((b.pmin.x == pmin.x && b.pmax.x == pmax.x) &&
				(b.pmin.y == pmin.y && b.pmax.y == pmax.y) &&
				(b.pmin.z == pmin.z && b.pmax.z == pmax.z));
	}

	__host__ __device__ bool operator!=(const HBoundingBox &b) const {
		return ((b.pmin.x != pmin.x || b.pmax.x != pmax.x) ||
				(b.pmin.y != pmin.y || b.pmax.y != pmax.y) ||
				(b.pmin.z != pmin.z || b.pmax.z != pmax.z));
	}

	__host__ __device__ glm::vec3 Diagonal() const { return pmax - pmin; }

	__host__ __device__ glm::vec3 Centroid() const { return 0.5f*(pmax + pmin); }

	__host__ __device__ float SurfaceArea() const {
		glm::vec3 d = Diagonal();
		return 2 * (d.x * d.y + d.y * d.z + d.z * d.x);
	}

	__host__ __device__ float Volume() const {
		glm::vec3 d = Diagonal();
		return d.x * d.y * d.z;
	}

	__host__ __device__ int MaximumDimension() const {
		glm::vec3 d = Diagonal();
		return (d.x > d.y && d.x > d.z) ? 0 : ((d.y > d.z) ? 1 : 2);
	}

	__host__ __device__ glm::vec3 Offset(const glm::vec3 &p) const {
		glm::vec3 o = p - pmin;
		if (pmax.x > pmin.x) o.x /= pmax.x - pmin.x;
		if (pmax.y > pmin.y) o.y /= pmax.y - pmin.y;
		if (pmax.z > pmin.z) o.z /= pmax.z - pmin.z;
		return o;
	}

	__host__ __device__ bool Intersect(const HRay &ray, float* t0, float* t1) const;

	glm::vec3 pmin;
	glm::vec3 pmax;
};	

//////////////////////////////////////////////////////////////////////////
// Geometry inline functions
//////////////////////////////////////////////////////////////////////////

__host__ __device__ inline HBoundingBox UnionP(const HBoundingBox &b, const glm::vec3 &p) {
	return HBoundingBox(glm::vec3(fmin(b.pmin.x, p.x),
								  fmin(b.pmin.y, p.y),
								  fmin(b.pmin.z, p.z)),
						glm::vec3(fmax(b.pmax.x, p.x),
								  fmax(b.pmax.y, p.y),
								  fmax(b.pmax.z, p.z)));
}

__host__ __device__ inline HBoundingBox UnionB(const HBoundingBox &b1, const HBoundingBox &b2) {
	return HBoundingBox(glm::vec3(fmin(b1.pmin.x, b2.pmin.x),
								  fmin(b1.pmin.y, b2.pmin.y),
								  fmin(b1.pmin.z, b2.pmin.z)),
						glm::vec3(fmax(b1.pmax.x, b2.pmax.x),
								  fmax(b1.pmax.y, b2.pmax.y),
								  fmax(b1.pmax.z, b2.pmax.z)));
}

__host__ __device__ inline HBoundingBox Intersection(const HBoundingBox &b1, const HBoundingBox &b2) {
	return HBoundingBox(glm::vec3(fmax(b1.pmin.x, b2.pmin.x),
								  fmax(b1.pmin.y, b2.pmin.y),
								  fmax(b1.pmin.z, b2.pmin.z)),
						glm::vec3(fmin(b1.pmax.x, b2.pmax.x),
								  fmin(b1.pmax.y, b2.pmax.y),
								  fmin(b1.pmax.z, b2.pmax.z)));
}

__host__ __device__ inline bool Overlaps(const HBoundingBox &b1, const HBoundingBox &b2) {
	bool x = (b1.pmax.x >= b2.pmin.x) && (b1.pmin.x <= b2.pmax.x);
	bool y = (b1.pmax.y >= b2.pmin.y) && (b1.pmin.y <= b2.pmax.y);
	bool z = (b1.pmax.z >= b2.pmin.z) && (b1.pmin.z <= b2.pmax.z);
	return (x && y && z);
}

__host__ __device__ inline bool Contains(const HBoundingBox &b, const glm::vec3 &p) {
	return (p.x >= b.pmin.x && p.x <= b.pmax.x &&
			p.y >= b.pmin.y && p.y <= b.pmax.y &&
			p.z >= b.pmin.z && p.z <= b.pmax.z);

}

__host__ __device__ inline void BoundingSphere(const HBoundingBox &b, glm::vec3* position, float* radius) {
	*position = (b.pmin + b.pmax) * 0.5f;
	*radius = Contains(b, *position) ? length(*position - b.pmax) : 0;
}

__host__ __device__ inline bool ContainsExclusive(const HBoundingBox &b, const glm::vec3 &p) {
	return (p.x >= b.pmin.x && p.x < b.pmax.x &&
			p.y >= b.pmin.y && p.y < b.pmax.y &&
			p.z >= b.pmin.z && p.z < b.pmax.z);
}

__host__ __device__ inline HBoundingBox Expand(const HBoundingBox &b, const float delta) {
	return HBoundingBox(b.pmin - glm::vec3(delta),
						b.pmax + glm::vec3(delta));
}

__host__ __device__ inline bool HBoundingBox::Intersect(const HRay &ray, float* t0, float* t1) const {
	
	return false;
}

#endif // GEOMETRY_H
