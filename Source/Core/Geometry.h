#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cuda_runtime.h>
#include <math.h>

class Vector3Df
{
public:
	union {
		struct { float x, y, z; };
		float _v[3];
	};

	__host__ __device__ Vector3Df(float _x = 0, float _y = 0, float _z = 0) : x(_x), y(_y), z(_z) {}
	__host__ __device__ Vector3Df(const Vector3Df& v) : x(v.x), y(v.y), z(v.z) {}
	__host__ __device__ Vector3Df(const float4& v) : x(v.x), y(v.y), z(v.z) {}
	inline __host__ __device__ float length(){ return sqrtf(x*x + y*y + z*z); }
	inline __host__ __device__ float lengthsq(){ return x*x + y*y + z*z; }
	inline __host__ __device__ void normalize(){ float norm = sqrtf(x*x + y*y + z*z); x /= norm; y /= norm; z /= norm; }
	inline __host__ __device__ Vector3Df& operator+=(const Vector3Df& v){ x += v.x; y += v.y; z += v.z; return *this; }
	inline __host__ __device__ Vector3Df& operator-=(const Vector3Df& v){ x -= v.x; y -= v.y; z -= v.z; return *this; }
	inline __host__ __device__ Vector3Df& operator*=(const float& a){ x *= a; y *= a; z *= a; return *this; }
	inline __host__ __device__ Vector3Df& operator*=(const Vector3Df& v){ x *= v.x; y *= v.y; z *= v.z; return *this; }
	inline __host__ __device__ Vector3Df operator*(float a) const{ return Vector3Df(x*a, y*a, z*a); }
	inline __host__ __device__ Vector3Df operator/(float a) const{ return Vector3Df(x / a, y / a, z / a); }
	inline __host__ __device__ Vector3Df operator*(const Vector3Df& v) const{ return Vector3Df(x * v.x, y * v.y, z * v.z); }
	inline __host__ __device__ Vector3Df operator+(const Vector3Df& v) const{ return Vector3Df(x + v.x, y + v.y, z + v.z); }
	inline __host__ __device__ Vector3Df operator-(const Vector3Df& v) const{ return Vector3Df(x - v.x, y - v.y, z - v.z); }
	inline __host__ __device__ Vector3Df& operator/=(const float& a){ x /= a; y /= a; z /= a; return *this; }
	inline __host__ __device__ bool operator!=(const Vector3Df& v){ return x != v.x || y != v.y || z != v.z; }
};

inline __host__ __device__ Vector3Df operator*(float a, Vector3Df v) { return Vector3Df(v.x*a, v.y*a, v.z*a); }
inline __host__ __device__ Vector3Df min3(const Vector3Df& v1, const Vector3Df& v2){ return Vector3Df(v1.x < v2.x ? v1.x : v2.x, v1.y < v2.y ? v1.y : v2.y, v1.z < v2.z ? v1.z : v2.z); }
inline __host__ __device__ Vector3Df max3(const Vector3Df& v1, const Vector3Df& v2){ return Vector3Df(v1.x > v2.x ? v1.x : v2.x, v1.y > v2.y ? v1.y : v2.y, v1.z > v2.z ? v1.z : v2.z); }
inline __host__ __device__ Vector3Df cross(const Vector3Df& v1, const Vector3Df& v2){ return Vector3Df(v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x); }
inline __host__ __device__ float dot(const Vector3Df& v1, const Vector3Df& v2){ return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
inline __host__ __device__ float dot(const Vector3Df& v1, const float4& v2){ return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
inline __host__ __device__ float dot(const float4& v1, const Vector3Df& v2){ return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }
inline __host__ __device__ float distancesq(const Vector3Df& v1, const Vector3Df& v2){ return (v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y) + (v1.z - v2.z)*(v1.z - v2.z); }
inline __host__ __device__ float distance(const Vector3Df& v1, const Vector3Df& v2){ return sqrtf((v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y) + (v1.z - v2.z)*(v1.z - v2.z)); }

class HRay
{
public:
	float3 Origin;
	float3 Direction;

	__device__
	HRay(float3 o_, float3 d_) : Origin(o_), Direction(d_) {}
};

#endif // GEOMETRY_H
