#ifndef INCLUDE_H
#define INCLUDE_H

//////////////////////////////////////////////////////////////////////////
// Global project includes
//////////////////////////////////////////////////////////////////////////

#include <cuda.h>
#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif // GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <GL\glew.h>
#include <GL\GL.h>
#include <GLFW\glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_math.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <vector>
#include <string>
#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#endif // INCLUDE_H