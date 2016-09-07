#ifndef MATERIAL_H
#define MATERIAL_H

#include <Core/Medium.h>

enum HMaterialType {// PBRT material types, only using a few atm
	//want to implement multiple B(R/T)DFs in the future
	REFLECTION		= 1<<0,
	TRANSMISSION	= 1<<1,
	DIFFUSE			= 1<<2,
	GLOSSY			= 1<<3,
	SPECULAR		= 1<<4,
	ALL_TYPES		= DIFFUSE | GLOSSY | SPECULAR,
	ALL_REFL		= REFLECTION | ALL_TYPES,
	ALL_TRANS		= TRANSMISSION | ALL_TYPES,
	ALL				= ALL_REFL | ALL_TRANS
};

struct HMaterial {
	
	__host__ __device__ HMaterial() {}
	__host__ __device__ HMaterial(glm::vec3 diff, glm::vec3 em)
		: diffuse(diff), emission(em) {}
	__host__ __device__ HMaterial(glm::vec3 diff, glm::vec3 em, glm::vec3 spec)
		: diffuse(diff), emission(em), specular(spec) {}
	__host__ __device__ HMaterial(glm::vec3 diff, glm::vec3 em, glm::vec3 spec, float eta)
		: diffuse(diff), emission(em), specular(spec), medium(HMedium(eta)) {}

	// TODO: Color, Scattering properties, BSDF etc...
	glm::vec3 diffuse;
	glm::vec3 specular;
	glm::vec3 emission;
	HMedium medium;
	HMaterialType materialType;

};

#endif // MATERIAL_H
