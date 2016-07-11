#include <cuda.h>
#include <cuda_runtime.h>
#include "GL\glew.h"
#include "GL\glut.h"
#include <cuda_gl_interop.h>
#include <sstream>
#include <cmath>
#include <math.h>

#include "Core/Camera.h"
#include "Core/Renderer.h"
#include "Core/Image.h"

//////////////////////////////////////////////////////////////////////////
// Constants
//////////////////////////////////////////////////////////////////////////
#define WINDOW_TITLE "OpenGL Window"
unsigned int WINDOW_WIDTH = 1280;
unsigned int WINDOW_HEIGHT = 720;
float FIELD_OF_VIEW = 45;

#define TEXTURE_ID 13

//////////////////////////////////////////////////////////////////////////
// Pointers
//////////////////////////////////////////////////////////////////////////
Camera* CameraObject = nullptr;
Renderer* RendererObject = nullptr;


//////////////////////////////////////////////////////////////////////////
// Function declarations
//////////////////////////////////////////////////////////////////////////
void InitCamera();
bool InitGL(int argc, char** argv);
bool InitCUDA(int argc, char** argv);
void Initialize(int argc, char **argv);

//////////////////////////////////////////////////////////////////////////
// User interaction callback declarations
//////////////////////////////////////////////////////////////////////////
void Display();
void Keyboard(unsigned char Key, int, int);
void SpecialKeys(int Key, int, int);
void Mouse(int Button, int State, int x, int y);
void Motion(int x, int y);

//////////////////////////////////////////////////////////////////////////
// Main loop
//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	// Main initialization
	Initialize(argc, argv);

	// Rendering main loop
	glutMainLoop();

}

//////////////////////////////////////////////////////////////////////////
// Camera initialization
//////////////////////////////////////////////////////////////////////////
void InitCamera()
{

	if (CameraObject)
	{
		delete CameraObject;
	}

	CameraObject = new Camera();
	CameraObject->SetResolution(WINDOW_WIDTH, WINDOW_HEIGHT);
	CameraObject->SetFOV(FIELD_OF_VIEW);

}

//////////////////////////////////////////////////////////////////////////
// Main initialization call
//////////////////////////////////////////////////////////////////////////
void Initialize(int argc, char** argv)
{

	InitCamera();

	// Initialize GL
	if (!InitGL(argc, argv))
	{
		return;
	}

	// Initialize CUDA
	InitCUDA(argc, argv);

	// Graphics display callback registration
	glutDisplayFunc(Display);

	// User interaction callback registration
	glutKeyboardFunc(Keyboard);
	glutSpecialFunc(SpecialKeys);
	glutMouseFunc(Mouse);
	glutMotionFunc(Motion);
	

}

//////////////////////////////////////////////////////////////////////////
// OpenGL initialization
//////////////////////////////////////////////////////////////////////////
bool InitGL(int argc, char** argv)
{

	// Create GL environment
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutCreateWindow(WINDOW_TITLE);

	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 "
		"GL_ARB_pixel_buffer_object"
		)) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}
	
	// Set up viewport
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	// Create a texture for display
	glBindTexture(GL_TEXTURE_2D, TEXTURE_ID);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
	
	// Enable texture
	glEnable(GL_TEXTURE_2D);
	
	return true;

}

//////////////////////////////////////////////////////////////////////////
// CUDA Initialization
//////////////////////////////////////////////////////////////////////////
bool InitCUDA(int argc, char** argv)
{
	return true;
}


//////////////////////////////////////////////////////////////////////////
// Callback functions
//////////////////////////////////////////////////////////////////////////
void Display()
{
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Texture refresh
	int ImageWidth = WINDOW_WIDTH;
	int ImageHeight = WINDOW_HEIGHT;
	uchar3* ImageData = new uchar3[ImageWidth*ImageHeight];

	for (int i = 0; i < ImageHeight; i++)
	{
		for (int j = 0; j < ImageWidth; j++)
		{
			// Remove, renders random colors.
			uchar3 temp;
			temp.x = (int) 255* rand() / RAND_MAX;
			temp.y = (int)255 * rand() / RAND_MAX;
			temp.z = (int)255 * rand() / RAND_MAX;
			ImageData[i*ImageWidth + j] = temp;
		}
	}

	glTexImage2D(GL_TEXTURE_2D,
		0,
		GL_RGB,
		ImageWidth,
		ImageHeight,
		0,
		GL_RGB,
		GL_UNSIGNED_BYTE,
		ImageData);
	delete[] ImageData;

	// Display texture
	glBindTexture(GL_TEXTURE_2D, TEXTURE_ID);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0);
	glVertex3f(0.0, 1.0, 0.0);
	glTexCoord2f(1.0, 0.0);
	glVertex3f(1.0, 1.0, 0.0);
	glTexCoord2f(1.0, 1.0);
	glVertex3f(1.0, 0.0, 0.0);
	glTexCoord2f(0.0, 1.0);
	glVertex3f(0.0, 0.0, 0.0);
	glEnd();

	glutSwapBuffers();
	glutPostRedisplay();
	
}

void Keyboard(unsigned char Key, int, int)
{

}

void SpecialKeys(int Key, int, int)
{

}

void Mouse(int Button, int State, int x, int y)
{

}

void Motion(int x, int y)
{

}
