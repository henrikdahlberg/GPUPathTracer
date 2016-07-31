#ifndef MATERIAL_H
#define MATERIAL_H

// TODO: Store predefined materials. Enums structure?
// The materials should be stored in a folder with one file each or defined in a text file
// When the Scene file is read, the materials data can then be retrieved from these files to instantiate the structs

struct HMaterial
{
	// TODO: Color, Scattering properties, BSDF etc...
	float3 Diffuse;
	float3 Emissive;
};

#endif // MATERIAL_H
