#ifndef MATERIAL_H
#define MATERIAL_H

struct HMaterial {
	
	__host__ __device__ HMaterial() {}
	__host__ __device__ HMaterial(glm::vec3 diff, glm::vec3 em)
		: diffuse(diff), emission(em) {}

	// TODO: Color, Scattering properties, BSDF etc...
	glm::vec3 diffuse;
	glm::vec3 emission;
};

#endif // MATERIAL_H
