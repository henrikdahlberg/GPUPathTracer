#include <cuda.h>
#include <cuda_runtime.h>
#include "GL\glew.h"
#include "GL\glut.h"
#include <cuda_gl_interop.h>
#include <iostream>
#include <sstream>
#include <cmath>
#include <math.h>

#include "Core/Scene.h"
#include "Core/Camera.h"
#include "Core/Renderer.h"
#include "Core/Image.h"

//////////////////////////////////////////////////////////////////////////
// Constants
//////////////////////////////////////////////////////////////////////////
#define WINDOW_TITLE_PREFIX "OpenGL Window"
unsigned int WINDOW_WIDTH = 1280;
unsigned int WINDOW_HEIGHT = 720;
unsigned int WINDOW_HANDLE = 0;
unsigned int FRAME_COUNT = 0;
float FIELD_OF_VIEW = 45;

//////////////////////////////////////////////////////////////////////////
// Pointers
//////////////////////////////////////////////////////////////////////////
HScene* Scene = nullptr;
HCamera* Camera = nullptr;
HRenderer* Renderer = nullptr;
HImage* FinalImage = nullptr;


//////////////////////////////////////////////////////////////////////////
// Function declarations
//////////////////////////////////////////////////////////////////////////
void InitCamera();
void InitGL(int argc, char** argv);
void InitCUDA(int argc, char** argv);
void Initialize(int argc, char** argv);

//////////////////////////////////////////////////////////////////////////
// OpenGL callback declarations
//////////////////////////////////////////////////////////////////////////
void Display();
void Reshape(int, int);
void Timer(int);
void Idle(void);
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

	// TODO: Move inside Initialize
	Renderer = new HRenderer(Camera);


	// Kernel call testing
	float* d_in;
	float* d_out;
	float hptr[10];
	for (int i = 0; i < 10; i++)
	{
		hptr[i] = i;
	}

	cudaMalloc((float**)&d_in, 10 * sizeof(float));
	cudaMalloc((float**)&d_out, 10 * sizeof(float));
	cudaMemcpy(d_in, hptr, 10 * sizeof(float), cudaMemcpyHostToDevice);
	Renderer->TestRunKernel(d_in, d_out);
	cudaMemcpy(hptr, d_out, 10 * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 10; i++)
	{
		std::cout << hptr[i] << std::endl;
	}

	// Rendering main loop
	glutMainLoop();

}

//////////////////////////////////////////////////////////////////////////
// Camera initialization
//////////////////////////////////////////////////////////////////////////
void InitCamera()
{

	if (Camera)
	{
		delete Camera;
	}

	Camera = new HCamera();
	Camera->SetResolution(WINDOW_WIDTH, WINDOW_HEIGHT);
	Camera->SetFOV(FIELD_OF_VIEW);

	if (!Camera)
	{
		fprintf(
			stderr,
			"ERROR: Failed Camera initialization.\n"
			);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

}

//////////////////////////////////////////////////////////////////////////
// Main initialization call
//////////////////////////////////////////////////////////////////////////
void Initialize(int argc, char** argv)
{

	InitCamera();

	// Initialize GL
	InitGL(argc, argv);

	// Initialize CUDA
	InitCUDA(argc, argv);

	// OpenGL callback registration
	glutDisplayFunc(Display);
	glutReshapeFunc(Reshape);
	glutIdleFunc(Idle);
	glutTimerFunc(0,Timer,0);
	glutKeyboardFunc(Keyboard);
	glutSpecialFunc(SpecialKeys);
	glutMouseFunc(Mouse);
	glutMotionFunc(Motion);

}

//////////////////////////////////////////////////////////////////////////
// OpenGL initialization
//////////////////////////////////////////////////////////////////////////
void InitGL(int argc, char** argv)
{

	// Create GL environment
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	WINDOW_HANDLE = glutCreateWindow(WINDOW_TITLE_PREFIX);

	if (WINDOW_HANDLE < 1)
	{
		fprintf(
			stderr,
			"ERROR: glutCreateWindow failed.\n"
			);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

	GLenum GLEW_INIT_RESULT;
	GLEW_INIT_RESULT = glewInit();
	if (GLEW_OK != GLEW_INIT_RESULT)
	{
		fprintf(
			stderr,
			"ERROR: %s\n",
			glewGetErrorString(GLEW_INIT_RESULT)
			);
		exit(EXIT_FAILURE);
	}

	if (!glewIsSupported("GL_VERSION_2_0 ""GL_ARB_pixel_buffer_object"))
	{
		fprintf(
			stderr,
			"ERROR: Support for necessary OpenGL extensions missing."
			);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}
	
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glDisable(GL_DEPTH_TEST);

	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

}

//////////////////////////////////////////////////////////////////////////
// CUDA Initialization
//////////////////////////////////////////////////////////////////////////
void InitCUDA(int argc, char** argv)
{

	if (false)
	{
		fprintf(
			stderr,
			"ERROR: Failed CUDA initialization."
			);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

}


//////////////////////////////////////////////////////////////////////////
// Callback functions
//////////////////////////////////////////////////////////////////////////
void Display()
{

	++FRAME_COUNT;
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	FinalImage = Renderer->Render();



	glutSwapBuffers();
	
}

void Reshape(int NewWidth, int NewHeight)
{

	WINDOW_WIDTH = NewWidth;
	WINDOW_HEIGHT = NewHeight;
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

}

void Timer(int Value)
{

	if (Value != 0)
	{
		char* WINDOW_TITLE = (char*)malloc(512 + strlen(WINDOW_TITLE_PREFIX));

		sprintf(
			WINDOW_TITLE,
			"%s: %d FPS @ %d x %d",
			WINDOW_TITLE_PREFIX,
			FRAME_COUNT * 5,
			WINDOW_WIDTH,
			WINDOW_HEIGHT
			);

		glutSetWindowTitle(WINDOW_TITLE);
		free(WINDOW_TITLE);
	}

	FRAME_COUNT = 0;
	glutTimerFunc(200, Timer, 1);

}

void Idle(void)
{
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
