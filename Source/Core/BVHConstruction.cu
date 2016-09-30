#include <Core/Include.h>
#include <Core/BVH.h>
#include <Shapes/Triangle.h>

// Define this to turn on error checking
//#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
				file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

inline void __cudaCheckError(const char *file, const int line) {
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
				file, line, cudaGetErrorString(err));
		exit(-1);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
				file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}


#define BLOCK_SIZE 128;

//////////////////////////////////////////////////////////////////////////
// Device functions
//////////////////////////////////////////////////////////////////////////

__device__ void CheckTreeRecursive(BVHNode* currentNode) {
	BVHNode* leftChild = currentNode->leftChild;
	BVHNode* rightChild = currentNode->rightChild;


}

/**
* Longest common prefix for Morton code
*/
__device__ int LongestCommonPrefix(int i, int j, int numTriangles,
								   MortonCode* mortonCodes, int* triangleIDs) {
	if (i < 0 || i > numTriangles - 1 || j < 0 || j > numTriangles - 1) {
		return -1;
	}

	MortonCode mi = mortonCodes[i];
	MortonCode mj = mortonCodes[j];

	if (mi == mj) {
		return __clzll(mi ^ mj) + __clzll(triangleIDs[i] ^ triangleIDs[j]);
	}
	else {
		return __clzll(mi ^ mj);
	}
}

/**
* Expand bits, used in Morton code calculation
*/
__device__ MortonCode bitExpansion(MortonCode i) {
	i = (i * 0x00010001u) & 0xFF0000FFu;
	i = (i * 0x00000101u) & 0x0F00F00Fu;
	i = (i * 0x00000011u) & 0xC30C30C3u;
	i = (i * 0x00000005u) & 0x49249249u;
	return i;
}

/**
* Compute morton code given triangle centroid scaled to [0,1] of scene bounding box
*/
__device__ MortonCode ComputeMortonCode(float x, float y, float z) {

	x = min(max(x * 1024.0f, 0.0f), 1023.0f);
	y = min(max(y * 1024.0f, 0.0f), 1023.0f);
	z = min(max(z * 1024.0f, 0.0f), 1023.0f);
	MortonCode xx = bitExpansion((MortonCode)x);
	MortonCode yy = bitExpansion((MortonCode)y);
	MortonCode zz = bitExpansion((MortonCode)z);
	return xx * 4 + yy * 2 + zz;

}

//////////////////////////////////////////////////////////////////////////
// Kernels
//////////////////////////////////////////////////////////////////////////

__global__ void ConstructBVH(BVHNode* BVHNodes, BVHNode* BVHLeaves,
							 int* nodeCounter,
							 HTriangle* triangles,
							 int* triangleIDs,
							 int numTriangles) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numTriangles) {
		BVHNode* leaf = BVHLeaves + i;

		int triangleIdx = triangleIDs[i];
		HTriangle triangle = triangles[triangleIdx];

		// Handle leaf first
		leaf->triangleIdx = triangleIdx;
		//printf("%d, %d\n", leaf->triangleIdx, (BVHLeaves + i)->triangleIdx);
		leaf->boundingBox = triangle.Bounds();

		BVHNode* current = leaf->parent;
		int currentIndex = current - BVHNodes;
		
		int res = atomicAdd(nodeCounter + currentIndex, 1);

		// Go up and handle internal nodes
		while (true) {
			if (res == 0) {
				return;
			}
			HBoundingBox leftBoundingBox = current->leftChild->boundingBox;
			HBoundingBox rightBoundingBox = current->rightChild->boundingBox;

			// Compute current bounding box
			current->boundingBox = UnionB(leftBoundingBox,
										  rightBoundingBox);

			// If current is root, return
			if (current == BVHNodes) {
				return;
			}
			current = current->parent;
			currentIndex = current - BVHNodes;
			res = atomicAdd(nodeCounter + currentIndex, 1);
		}
	}
}

/**
* Radix tree construction kernel
* Algorithm described in karras2012 paper.
* Node-wise parallel
*/
__global__ void BuildRadixTree(BVHNode* radixTreeNodes,
							   BVHNode* radixTreeLeaves,
							   MortonCode* mortonCodes,
							   int* triangleIDs,
							   int numTriangles) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numTriangles - 1) {

		// Run radix tree construction algorithm
		// Determine direction of the range (+1 or -1)
		int d = LongestCommonPrefix(i, i + 1, numTriangles, mortonCodes, triangleIDs) -
			LongestCommonPrefix(i, i - 1, numTriangles, mortonCodes, triangleIDs) >= 0 ? 1 : -1;

		// Compute upper bound for the length of the range
		int deltaMin = LongestCommonPrefix(i, i - d, numTriangles, mortonCodes, triangleIDs);
		//int lmax = 128;
		int lmax = 2;

		while (LongestCommonPrefix(i, i + lmax*d, numTriangles, mortonCodes, triangleIDs) > deltaMin) {
			//lmax = lmax * 4;
			lmax = lmax * 2;
		}

		// Find the other end using binary search
		int l = 0;
		int divider = 2;
		for (int t = lmax / divider; t >= 1; divider *= 2) {
			if (LongestCommonPrefix(i, i + (l + t) * d, numTriangles, mortonCodes, triangleIDs) > deltaMin) {
				l = l + t;
			}
			if (t == 1) break;
			t = lmax / divider;
		}

		int j = i + l * d;

		// Find the split position using binary search
		int deltaNode = LongestCommonPrefix(i, j, numTriangles, mortonCodes, triangleIDs);
		int s = 0;
		divider = 2;
		for (int t = (l + (divider - 1)) / divider; t >= 1; divider *= 2) {
			if (LongestCommonPrefix(i, i + (s + t) * d, numTriangles, mortonCodes, triangleIDs) > deltaNode) {
				s = s + t;
			}
			if (t == 1) break;
			t = (l + (divider - 1)) / divider;
		}

		int gamma = i + s * d + min(d, 0);

		//printf("i:%d, d:%d, deltaMin:%d, deltaNode:%d, lmax:%d, l:%d, j:%d, gamma:%d. \n", i, d, deltaMin, deltaNode, lmax, l, j, gamma);

		// Output child pointers
		BVHNode* current = radixTreeNodes + i;

		if (min(i, j) == gamma) {
			current->leftChild = radixTreeLeaves + gamma;
			(radixTreeLeaves + gamma)->parent = current;
		}
		else {
			current->leftChild = radixTreeNodes + gamma;
			(radixTreeNodes + gamma)->parent = current;
		}

		if (max(i, j) == gamma + 1) {
			current->rightChild = radixTreeLeaves + gamma + 1;
			(radixTreeLeaves + gamma + 1)->parent = current;
		}
		else {
			current->rightChild = radixTreeNodes + gamma + 1;
			(radixTreeNodes + gamma + 1)->parent = current;
		}

		current->minId = min(i, j);
		current->maxId = max(i, j);
	}
}

__global__ void ComputeBoundingBoxes(HTriangle* triangles,
									 int numTriangles,
									 HBoundingBox* boundingBoxes) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numTriangles) {
		HTriangle triangle = triangles[i];
		//triangle.boundingBox = UnionP(HBoundingBox(triangle.v0, triangle.v1), triangle.v2);
		//boundingBoxes[i] = triangle.boundingBox;
		boundingBoxes[i] = triangle.Bounds();
		//triangles[i] = triangle;
	}
}

__global__ void ComputeMortonCodes(HTriangle* triangles,
								   int numTriangles,
								   HBoundingBox sceneBounds,
								   MortonCode* mortonCodes) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numTriangles) {
		HTriangle triangle = triangles[i];

		// Compute triangle centroid
		float div = 1.0f / 3.0f;
		glm::vec3 centroid = (triangle.v0 + triangle.v1 + triangle.v2)*div;

		// Normalize triangle centroid to lie within [0,1] of scene bounding box
		float x = (centroid.x - sceneBounds.pmin.x) / (sceneBounds.pmax.x - sceneBounds.pmin.x);
		float y = (centroid.y - sceneBounds.pmin.y) / (sceneBounds.pmax.y - sceneBounds.pmin.y);
		float z = (centroid.z - sceneBounds.pmin.z) / (sceneBounds.pmax.z - sceneBounds.pmin.z);

		// Compute morton code
		mortonCodes[i] = ComputeMortonCode(x, y, z);
	}
}

__global__ void DebugBVH(BVHNode* BVHLeaves, BVHNode* BVHNodes, int numTriangles) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	//if (i < numTriangles) {
	//	//printf("%d\n", (BVHLeaves + i)->triangleIdx);
	//	bool broken = false;
	//	BVHNode* current = BVHLeaves + i;
	//	BVHNode* last;
	//	while (true) {
	//		last = current;
	//		current = current->parent;

	//		if (last == rootNode && current == nullptr) {
	//			break;
	//		}

	//		broken = (current->leftChild != last && current->rightChild != last);
	//		if (broken) {
	//			printf("BVH broken");
	//		}

	//	}

	//}

	// do in serial
	if (i == 0) {
		for (int j = 0; j < numTriangles; j++) {
			BVHNode* currentNode = BVHLeaves + j;
			printf("BBox for triangleIdx %d: pmin: (%f,%f,%f), pmax: (%f,%f,%f)\n",
				   (BVHLeaves + j)->triangleIdx,
				   currentNode->boundingBox.pmin.x,
				   currentNode->boundingBox.pmin.y,
				   currentNode->boundingBox.pmin.z,
				   currentNode->boundingBox.pmax.x,
				   currentNode->boundingBox.pmax.y,
				   currentNode->boundingBox.pmax.z);
		}
		//parents:
		for (int j = 0; j < numTriangles; j++) {
			BVHNode* currentNode = (BVHLeaves + j)->parent;
			printf("BBox for parent node of triangleIdx %d: pmin: (%f,%f,%f), pmax: (%f,%f,%f)\n",
				   (BVHLeaves + j)->triangleIdx,
				   currentNode->boundingBox.pmin.x,
				   currentNode->boundingBox.pmin.y,
				   currentNode->boundingBox.pmin.z,
				   currentNode->boundingBox.pmax.x,
				   currentNode->boundingBox.pmax.y,
				   currentNode->boundingBox.pmax.z);
		}

		for (int j = 0; j < numTriangles; j++) {
			BVHNode* currentNode = (BVHLeaves + j)->parent->parent;
			printf("BBox for parents parent node of triangleIdx %d: pmin: (%f,%f,%f), pmax: (%f,%f,%f)\n",
				   (BVHLeaves + j)->triangleIdx,
				   currentNode->boundingBox.pmin.x,
				   currentNode->boundingBox.pmin.y,
				   currentNode->boundingBox.pmin.z,
				   currentNode->boundingBox.pmax.x,
				   currentNode->boundingBox.pmax.y,
				   currentNode->boundingBox.pmax.z);
		}

		for (int j = 0; j < numTriangles; j++) {
			BVHNode* currentNode = (BVHLeaves + j)->parent->parent->parent;
			printf("BBox for parents parents parent node of triangleIdx %d: pmin: (%f,%f,%f), pmax: (%f,%f,%f)\n",
				   (BVHLeaves + j)->triangleIdx,
				   currentNode->boundingBox.pmin.x,
				   currentNode->boundingBox.pmin.y,
				   currentNode->boundingBox.pmin.z,
				   currentNode->boundingBox.pmax.x,
				   currentNode->boundingBox.pmax.y,
				   currentNode->boundingBox.pmax.z);
		}

	}

	//if (i == 0) {
	//	for (int j = 0; j < numTriangles; j++) {
	//		//BVHNode* currentNode = BVHLeaves + j;
	//		BVHNode* currentNode = (BVHLeaves + j)->parent;

	//		int p = 1;
	//		//while (currentNode->parent != nullptr) {
	//		while (currentNode != nullptr) {
	//			/*if (currentNode->parent->leftChild != currentNode && currentNode->parent->rightChild != currentNode) {
	//				printf("\n SOMETHING IS WRONG\n");
	//				}*/
	//			if (currentNode->leftChild == nullptr) {
	//				printf("\n Parent %d from triangle %d is missing left child\n", p, j);
	//			}
	//			if (currentNode->rightChild == nullptr) {
	//				printf("\n Parent %d from triangle %d is missing right child\n", p, j);
	//			}
	//			currentNode = currentNode->parent;
	//			++p;
	//		}

	//	}
	//}
}

struct BoundingBoxUnion {
	__host__ __device__ HBoundingBox operator()(const HBoundingBox &b1, const HBoundingBox &b2) const {
		return UnionB(b1, b2);
	}
};

extern "C" void BuildBVH(BVH& bvh, HTriangle* triangles, int numTriangles, HBoundingBox &sceneBounds) {

	int blockSize = BLOCK_SIZE;
	int gridSize = (numTriangles + blockSize - 1) / blockSize;

	// Timing metrics
	float total = 0;
	float elapsed;
	cudaEvent_t start, stop;

	std::cout << "Number of triangles: " << numTriangles << std::endl;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Pre-process stage, scene bounding box
	// TODO: add check if this has been done already
	//		 if we already have scenebounds and have new/modified triangles, no need to start over
	// Should only do this if scene has changed (added tris, moved tris)

	// Compute bounding boxes
	std::cout << "Computing triangle bounding boxes...";
	cudaEventRecord(start, 0);
	thrust::device_vector<HBoundingBox> boundingBoxes(numTriangles);
	ComputeBoundingBoxes << <gridSize, blockSize >> >(triangles, numTriangles, boundingBoxes.data().get());
	CudaCheckError();
	checkCudaErrors(cudaGetLastError());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << " done! Computation took " << elapsed << " ms." << std::endl;
	total += elapsed;

	// Compute scene bounding box
	std::cout << "Computing scene bounding box...";
	cudaEventRecord(start, 0);
	sceneBounds = thrust::reduce(boundingBoxes.begin(), boundingBoxes.end(), HBoundingBox(), BoundingBoxUnion());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << " done! Computation took " << elapsed << " ms." << std::endl;
	total += elapsed;
	std::cout << "Total pre-computation time for scene was " << total << " ms.\n" << std::endl;
	total = 0;

	std::cout << "Scene boundingbox:\n";
	std::cout << "pmin: " << sceneBounds.pmin.x << ", " << sceneBounds.pmin.y << ", " << sceneBounds.pmin.z << std::endl;
	std::cout << "pmax: " << sceneBounds.pmax.x << ", " << sceneBounds.pmax.y << ", " << sceneBounds.pmax.z << std::endl;

	// Pre-process done, start building BVH

	// Compute Morton codes
	thrust::device_vector<MortonCode> mortonCodes(numTriangles);
	std::cout << "Computing Morton codes...";
	cudaEventRecord(start, 0);
	ComputeMortonCodes << <gridSize, blockSize >> >(triangles,
													numTriangles,
													sceneBounds,
													mortonCodes.data().get());
	CudaCheckError();
	checkCudaErrors(cudaGetLastError());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << " done! Computation took " << elapsed << " ms." << std::endl;
	total += elapsed;

	// Sort triangle indices with Morton code as key
	thrust::device_vector<int> triangleIDs(numTriangles);
	thrust::sequence(triangleIDs.begin(), triangleIDs.end());
	std::cout << "Sort triangles...";
	cudaEventRecord(start, 0);
	try {
		thrust::sort_by_key(mortonCodes.begin(), mortonCodes.end(), triangleIDs.begin());
	}
	catch (thrust::system_error e) {
		std::cout << "Error inside sort: " << e.what() << std::endl;
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << " done! Sorting took " << elapsed << " ms." << std::endl;
	total += elapsed;

	// Build radix tree of BVH nodes
	checkCudaErrors(cudaMalloc((void**)&bvh.BVHNodes, (numTriangles - 1)*sizeof(BVHNode)));
	checkCudaErrors(cudaMalloc((void**)&bvh.BVHLeaves, numTriangles*sizeof(BVHNode)));
	std::cout << "Building radix tree...";
	cudaEventRecord(start, 0);
	BuildRadixTree << <gridSize, blockSize >> >(bvh.BVHNodes,
												bvh.BVHLeaves,
												mortonCodes.data().get(),
												triangleIDs.data().get(),
												numTriangles);
	CudaCheckError();
	checkCudaErrors(cudaGetLastError());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << " done! Took " << elapsed << " ms." << std::endl;
	total += elapsed;

	// Build BVH
	thrust::device_vector<int> nodeCounters(numTriangles);
	std::cout << "Building BVH...";
	cudaEventRecord(start, 0);
	ConstructBVH << <gridSize, blockSize >> >(bvh.BVHNodes,
											  bvh.BVHLeaves,
											  nodeCounters.data().get(),
											  triangles,
											  triangleIDs.data().get(),
											  numTriangles);
	CudaCheckError();
	//checkCudaErrors(cudaDeviceSynchronize());
	//DebugBVH << <gridSize, blockSize >> >(bvh.BVHLeaves, bvh.BVHNodes, numTriangles);
	checkCudaErrors(cudaGetLastError());
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	std::cout << " done! Took " << elapsed << " ms." << std::endl;
	total += elapsed;

	std::cout << "Total BVH construction time was " << total << " ms.\n" << std::endl;
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}