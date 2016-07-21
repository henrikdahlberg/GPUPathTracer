#include "Image.h"

HImage::HImage()
{

}

HImage::HImage(unsigned int Width, unsigned int Height)
{
	Resolution.x = Width;
	Resolution.y = Height;
	NumPixels = Width*Height;
	Pixels = new Vector3Df[NumPixels];
}

HImage::HImage(float2 Res)
{
	Resolution = Res;
	NumPixels = Resolution.x*Resolution.y;
	Pixels = new Vector3Df[NumPixels];
}

HImage::~HImage()
{
	delete[] Pixels;
	Pixels = nullptr;
}
