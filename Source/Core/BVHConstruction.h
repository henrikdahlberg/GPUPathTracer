#ifndef BVHCONSTRUCTION_H
#define BVHCONSTRUCTION_H

#include <Core/BVH.h>
#include <Core/Geometry.h>
#include <Shapes/Triangle.h>

extern "C" void BuildBVH(BVH& bvh,
						 HTriangle* triangles,
						 int numTriangles,
						 HBoundingBox& sceneBounds);

#endif // BVHCONSTRUCTION_H