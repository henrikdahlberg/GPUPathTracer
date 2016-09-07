#ifndef MEDIUM_H
#define MEDIUM_H

#include <Core/Include.h>

struct HScatteringProperties {

	__host__ __device__ HScatteringProperties() {
		reducedScatteringCoefficient = 0.0f;
		absorptionMultiplier = glm::vec3(0.0f);
	}
	__host__ __device__ HScatteringProperties(float rsc, glm::vec3 abs)
		: reducedScatteringCoefficient(rsc), absorptionMultiplier(abs) {}

	float reducedScatteringCoefficient;
	glm::vec3 absorptionMultiplier;

};

struct HMedium {

	__host__ __device__ HMedium() {
		eta = 1.000293f; //default IOR of air
		scatteringProperties = HScatteringProperties();
	}
	__host__ __device__ HMedium(float e)
		: eta(e) {}
	__host__ __device__ HMedium(float e, HScatteringProperties sp)
		: eta(e), scatteringProperties(sp) {}
	__host__ __device__ HMedium(float e, float rsc, glm::vec3 abs)
		: eta(e), scatteringProperties(HScatteringProperties(rsc, abs)) {}

	float eta;
	HScatteringProperties scatteringProperties;

};


#endif