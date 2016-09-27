#include <Core/BVH.h>

bool BVH::IsLeaf(BVHNode* node) {
	return !node->leftChild || !node->rightChild;
}