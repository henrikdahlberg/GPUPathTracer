#include <cuda.h>
#include <cuda_runtime.h>
#include <GL\glew.h>
#include <GL\glut.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <sstream>
#include <cmath>
#include <math.h>

#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif // GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <Core/Scene.h>
#include <Core/Camera.h>
#include <Core/Renderer.h>
#include <Core/Image.h>

//////////////////////////////////////////////////////////////////////////
// Constants
//////////////////////////////////////////////////////////////////////////
#define WINDOW_TITLE_PREFIX "OpenGL Window"
#define FPS_DISPLAY_REFRESH_RATE 200 //ms
unsigned int WINDOW_WIDTH = 1280;
unsigned int WINDOW_HEIGHT = 720;
unsigned int WINDOW_HANDLE = 0;

//////////////////////////////////////////////////////////////////////////
// Pointers
//////////////////////////////////////////////////////////////////////////
HScene* scene = nullptr;
HCamera* camera = nullptr;
HRenderer* renderer = nullptr;
HImage* Image = nullptr;

//////////////////////////////////////////////////////////////////////////
// Function declarations
//////////////////////////////////////////////////////////////////////////
void InitCamera();
void InitGL(int argc, char** argv);
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
void Cleanup();

//////////////////////////////////////////////////////////////////////////
// Main loop
//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{

	// Main initialization
	Initialize(argc, argv);

	// TODO: Move inside Initialize
	scene = new HScene();
	scene->LoadSceneFile();
	renderer = new HRenderer(camera);
	renderer->InitScene(scene);

	// Rendering main loop
	glutMainLoop();

}

//////////////////////////////////////////////////////////////////////////
// Camera initialization
//////////////////////////////////////////////////////////////////////////
void InitCamera()
{
	
	if (camera)
	{
		delete camera;
	}

	camera = new HCamera(WINDOW_WIDTH, WINDOW_HEIGHT);

	if (!camera)
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

	// OpenGL callback registration
	glutDisplayFunc(Display);
	glutReshapeFunc(Reshape);
	glutIdleFunc(Idle);
	glutTimerFunc(0,Timer,0);
	glutKeyboardFunc(Keyboard);
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

	// Window creation error handling
	if (WINDOW_HANDLE < 1)
	{
		fprintf(
			stderr,
			"ERROR: glutCreateWindow failed.\n"
			);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}

	// GLEW initialization error handling
	GLenum GLEW_INIT_RESULT;
	GLEW_INIT_RESULT = glewInit();
	if (GLEW_INIT_RESULT != GLEW_OK)
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

	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, WINDOW_WIDTH, 0.0, WINDOW_HEIGHT, 1.0, -1.0);
	glMatrixMode(GL_MODELVIEW);

}

//////////////////////////////////////////////////////////////////////////
// Callback functions
//////////////////////////////////////////////////////////////////////////
void Display()
{

	renderer->fpsCounter++;
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	Image = renderer->Render();

	glBindBuffer(GL_ARRAY_BUFFER, Image->buffer);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, WINDOW_WIDTH*WINDOW_HEIGHT);
	glDisableClientState(GL_VERTEX_ARRAY);
	glutSwapBuffers();

}

void Reshape(int width, int height)
{

	WINDOW_WIDTH = width;
	WINDOW_HEIGHT = height;

	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, WINDOW_WIDTH, 0.0, WINDOW_HEIGHT, 1.0, -1.0);
	glMatrixMode(GL_MODELVIEW);

	camera->SetResolution(WINDOW_WIDTH, WINDOW_HEIGHT);
	renderer->Resize(camera->GetCameraData());
}

void Timer(int value)
{

	if (value != 0)
	{
		char* WINDOW_TITLE = (char*)malloc(512 + strlen(WINDOW_TITLE_PREFIX));

		sprintf(
			WINDOW_TITLE,
			"%s: %d FPS @ %d x %d, Iterations: %d",
			WINDOW_TITLE_PREFIX,
			renderer->fpsCounter * 1000 / FPS_DISPLAY_REFRESH_RATE,
			WINDOW_WIDTH,
			WINDOW_HEIGHT,
			renderer->passCounter);

		glutSetWindowTitle(WINDOW_TITLE);
		free(WINDOW_TITLE);
	}

	renderer->fpsCounter = 0;
	glutPostRedisplay();
	glutTimerFunc(FPS_DISPLAY_REFRESH_RATE, Timer, 1);

}

void Idle(void)
{
	glutPostRedisplay();
}

void Keyboard(unsigned char Key, int, int)
{

}

void Mouse(int button, int state, int x, int y)
{
	// TEMP
	scene->LoadSceneFile();
	renderer->Update(camera);
	renderer->InitScene(scene);

	Motion(x, y);

}

void Motion(int x, int y)
{

}
