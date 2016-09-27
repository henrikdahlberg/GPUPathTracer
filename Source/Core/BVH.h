#ifndef BVH_H
#define BVH_H

#include <Core/Include.h>
#include <Core/Geometry.h>

typedef unsigned long long MortonCode;

struct BVHNode {
	int minId;
	int maxId; // do we even use these?
	int triangleIdx;
	int splitDim;

	BVHNode* leftChild;
	BVHNode* rightChild;
	BVHNode* parent;

	HBoundingBox boundingBox;

};

class BVH {
public:
	BVH() {}
	virtual ~BVH() {}

	bool IsLeaf(BVHNode* node);
	BVHNode* GetRoot();

	BVHNode* BVHNodes;
	BVHNode* BVHLeaves;
	int numTriangles;
};

#endif // BVH_H