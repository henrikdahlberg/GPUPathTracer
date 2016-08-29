#include <Core/Include.h>

#include <Core/Scene.h>
#include <Core/Camera.h>
#include <Core/Renderer.h>
#include <Core/Image.h>

//////////////////////////////////////////////////////////////////////////
// Global variables
//////////////////////////////////////////////////////////////////////////
#define WINDOW_TITLE_PREFIX "OpenGL Window"
unsigned int WINDOW_WIDTH = 1024;
unsigned int WINDOW_HEIGHT = 1024;
double lastTime = 0.0;
double deltaTime;
bool input[1024]; //TODO: Some input controller class
bool bReset = false;
bool bRotateCamera = false;
glm::uvec2 lastMousePos = glm::uvec2(0, 0);

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
void SetWindowTitle(GLFWwindow* window, double deltaTime);
void Update();
void Display();
void ProcessMovement(double deltaTime);

//////////////////////////////////////////////////////////////////////////
// OpenGL callback declarations
//////////////////////////////////////////////////////////////////////////
void Resize(GLFWwindow* window, int width, int height);
void Keyboard(GLFWwindow* window, int key, int scancode, int action, int mode);
void Mouse(GLFWwindow* window, int button, int action, int mods);
void Motion(GLFWwindow* window, double xpos, double ypos);
void Scroll(GLFWwindow* window, double xoffset, double yoffset);

//////////////////////////////////////////////////////////////////////////
// Main loop
//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	// Main initialization
	Initialize(argc, argv);

	// Rendering main loop
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		Update();
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
	glfwSetWindowSizeCallback(window, Resize);
	glfwSetKeyCallback(window, Keyboard);
	glfwSetMouseButtonCallback(window, Mouse);
	glfwSetCursorPosCallback(window, Motion);
	glfwSetScrollCallback(window, Scroll);
}

//////////////////////////////////////////////////////////////////////////
// OpenGL initialization
//////////////////////////////////////////////////////////////////////////
void InitGL(int argc, char** argv) {

	// Create GL environment
	// TODO: Convert project to modern OpenGL
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
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

void Resize(GLFWwindow* window, int width, int height) {

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

void SetWindowTitle(GLFWwindow* window, double deltaTime) {
	char* WINDOW_TITLE = (char*)malloc(512 + strlen(WINDOW_TITLE_PREFIX));
	sprintf(WINDOW_TITLE,
			"%s: %.0f FPS @ %d x %d, Iterations: %d, Rendering time: %.0f ms",
			WINDOW_TITLE_PREFIX,
			1.0f / deltaTime,
			WINDOW_WIDTH,
			WINDOW_HEIGHT,
			renderer->passCounter,
			deltaTime * 1000.0f);

	glfwSetWindowTitle(window, WINDOW_TITLE);
}

void Keyboard(GLFWwindow* window, int key, int scancode, int action, int mode) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
	if (key == GLFW_KEY_PRINT_SCREEN && action == GLFW_PRESS) {
		image->SavePNG("Images/Screenshot");
	}
	if (key >= 0 && key < 1024) {
		switch (action) {
		case GLFW_PRESS:
			input[key] = true;	break;
		case GLFW_RELEASE:
			input[key] = false;
			bReset = false;		break;
		default:
			break;
		}
	}
}

void Mouse(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
		bRotateCamera = true;
	}
	if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) {
		bRotateCamera = false;
	}
}

void Motion(GLFWwindow* window, double xpos, double ypos) {
	float xoffset = xpos - lastMousePos.x;
	float yoffset = ypos - lastMousePos.y;

	lastMousePos.x = xpos;
	lastMousePos.y = ypos;

	if (bRotateCamera) {
		camera->ProcessMouseMovement(xoffset, yoffset);
		bReset = true;
	}
}

void Scroll(GLFWwindow* window, double xoffset, double yoffset) {
	camera->ProcessMouseScroll(yoffset);
}

void ProcessMovement(double deltaTime) {

	if (input[GLFW_KEY_W]) {
		camera->ProcessMovement(FORWARD, deltaTime);
		bReset = true;
	}
	if (input[GLFW_KEY_A]) {
		camera->ProcessMovement(LEFT, deltaTime);
		bReset = true;
	}
	if (input[GLFW_KEY_S]) {
		camera->ProcessMovement(BACKWARD, deltaTime);
		bReset = true;
	}
	if (input[GLFW_KEY_D]) {
		camera->ProcessMovement(RIGHT, deltaTime);
		bReset = true;
	}
	if (input[GLFW_KEY_SPACE]) {
		camera->ProcessMovement(UP, deltaTime);
		bReset = true;
	}
	if (input[GLFW_KEY_LEFT_CONTROL]) {
		camera->ProcessMovement(DOWN, deltaTime);
		bReset = true;
	}
	if (input[GLFW_KEY_R]) {
		scene->LoadSceneFile();
		renderer->InitScene(scene);
		bReset = true;
	}
}

void Update() {
	double currentTime = glfwGetTime();
	deltaTime = currentTime - lastTime;
	SetWindowTitle(window, deltaTime);
	ProcessMovement(deltaTime);
	if (bReset) {
		renderer->Reset(camera);
		bReset = false;
	}
	Display();
	lastTime = currentTime;
}
