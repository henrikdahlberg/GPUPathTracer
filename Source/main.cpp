#include <Core/Include.h>

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

//////////////////////////////////////////////////////////////////////////
// Pointers
//////////////////////////////////////////////////////////////////////////
HScene* scene = nullptr;
HCamera* camera = nullptr;
HRenderer* renderer = nullptr;
HImage* image = nullptr;
GLFWwindow* window = nullptr;

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
void Keyboard(GLFWwindow* window, int key, int scancode, int action, int mode);
void SpecialKeys(int Key, int, int);
void Mouse(int Button, int State, int x, int y);
void Motion(int x, int y);
void Cleanup();

//////////////////////////////////////////////////////////////////////////
// Main loop
//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	// Main initialization
	Initialize(argc, argv);

	// Rendering main loop
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		Display();
		glfwSwapBuffers(window);
	}

	glfwTerminate();
	return 0;
}

//////////////////////////////////////////////////////////////////////////
// Camera initialization
//////////////////////////////////////////////////////////////////////////
void InitCamera() {

	if (camera) {
		delete camera;
	}

	camera = new HCamera(WINDOW_WIDTH, WINDOW_HEIGHT);

	if (!camera) {
		fprintf(stderr, "ERROR: Failed Camera initialization.\n");
		fflush(stderr);
		exit(EXIT_FAILURE);
	}
}

//////////////////////////////////////////////////////////////////////////
// Main initialization call
//////////////////////////////////////////////////////////////////////////
void Initialize(int argc, char** argv) {

	InitCamera();

	// Initialize GL
	InitGL(argc, argv);

	scene = new HScene();
	scene->LoadSceneFile();
	renderer = new HRenderer(camera);
	renderer->InitScene(scene);

	// OpenGL callback registration
	glfwSetKeyCallback(window, Keyboard);
	//glutDisplayFunc(Display);
	//glutReshapeFunc(Reshape);
	//glutIdleFunc(Idle);
	//glutTimerFunc(0, Timer, 0);
	//glutKeyboardFunc(Keyboard);
	//glutMouseFunc(Mouse);
	//glutMotionFunc(Motion);
}

//////////////////////////////////////////////////////////////////////////
// OpenGL initialization
//////////////////////////////////////////////////////////////////////////
void InitGL(int argc, char** argv) {

	// Create GL environment
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	//glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT,
							  WINDOW_TITLE_PREFIX,
							  nullptr, nullptr);
	if (window == nullptr) {
		fprintf(stderr, "ERROR: glfwCreateWindow failed.\n");
		fflush(stderr);
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	GLenum GLEW_INIT_RESULT;
	GLEW_INIT_RESULT = glewInit();
	if (GLEW_INIT_RESULT != GLEW_OK) {
		fprintf(stderr, "GLEW initialization error: %s\n",
				glewGetErrorString(GLEW_INIT_RESULT));
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
void Display() {

	renderer->fpsCounter++;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	image = renderer->Render();

	glBindBuffer(GL_ARRAY_BUFFER, image->buffer);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, WINDOW_WIDTH*WINDOW_HEIGHT);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	// NSIGHT profiling, exit after one iteration
	//cudaDeviceSynchronize();
	//exit(EXIT_SUCCESS);
}

void Reshape(int width, int height) {

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

void Timer(int value) {

	// TODO: Validate that this is working as intended
	//		 weird frame rates shown when rendering is slow
	if (value != 0) {
		char* WINDOW_TITLE = (char*)malloc(512 + strlen(WINDOW_TITLE_PREFIX));

		sprintf(
			WINDOW_TITLE,
			"%s: %d FPS @ %d x %d, Iterations: %d",
			WINDOW_TITLE_PREFIX,
			renderer->fpsCounter * 1000 / FPS_DISPLAY_REFRESH_RATE,
			WINDOW_WIDTH,
			WINDOW_HEIGHT,
			renderer->passCounter);

		//glutSetWindowTitle(WINDOW_TITLE);
		free(WINDOW_TITLE);
	}

	renderer->fpsCounter = 0;
	//glutPostRedisplay();
	//glutTimerFunc(FPS_DISPLAY_REFRESH_RATE, Timer, 1);
}

void Idle(void) {
	//glutPostRedisplay();
}

void Keyboard(GLFWwindow* window, int key, int scancode, int action, int mode) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

void Mouse(int button, int state, int x, int y) {
	// TEMP, randomizes scene
	scene->LoadSceneFile();
	renderer->Update(camera);
	renderer->InitScene(scene);

	Motion(x, y);
}

void Motion(int x, int y) {

}
