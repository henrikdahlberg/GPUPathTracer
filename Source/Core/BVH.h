#ifndef BVH_H
#define BVH_H

#include <Core/Include.h>
#include <Core/Geometry.h>

typedef unsigned long long MortonCode;

struct BVHNode {

	__host__ __device__ inline bool IsLeaf() { return !leftChild && !rightChild; }

	int minId;
	int maxId; // do we even use these?
	int triangleIdx;

	BVHNode* leftChild;
	BVHNode* rightChild;
	BVHNode* parent;

	HBoundingBox boundingBox;

};

class BVH {
public:
	BVH() {}
	virtual ~BVH() {}

	BVHNode* GetRoot();

	BVHNode* BVHNodes;
	BVHNode* BVHLeaves;
	int numTriangles;
};

#endif // BVH_H