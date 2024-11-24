#include <cuda.h>
#include <cuda_runtime.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Geometry"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "imgui.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>

#define G 6.67408e-11f

int width = 1280;
int height = 720;

// Function to compile shader
GLuint compileShader(GLenum type, const char *source) {
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);

  GLint success;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    char infoLog[512];
    glGetShaderInfoLog(shader, 512, nullptr, infoLog);
    fprintf(stderr, "ERROR::SHADER::COMPILATION_FAILED\n%s\n", infoLog);
  }
  return shader;
}

// // Function to create shader program
GLuint createShaderProgram(const char *vertexSource,
                           const char *fragmentSource) {
  GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSource);
  GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSource);

  GLuint shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);

  GLint success;
  glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
  if (!success) {
    char infoLog[512];
    glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
    fprintf(stderr, "ERROR::PROGRAM::LINKING_FAILED\n%s\n", infoLog);
  }

  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  return shaderProgram;
}
void setupImGui(GLFWwindow *window) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  if (!ImGui_ImplGlfw_InitForOpenGL(window, true)) {
    fprintf(stderr, "Failed to initialize ImGui GLFW backend\n");
    return;
  }
  if (!ImGui_ImplOpenGL3_Init("#version 150")) {
    fprintf(stderr, "Failed to initialize ImGui OpenGL3 backend\n");
    return;
  }
}

void cleanupImGui() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}

// // shader to draw points
const char *vertexShaderSource = R"(
#version 150 core
uniform float aspectRatio;
in vec3 position;

void main() {
    vec3 adjustedPosition = position * 0.5;
    adjustedPosition.x /= aspectRatio; // Correct for aspect ratio
    gl_Position = vec4(adjustedPosition, 1.0);
    gl_PointSize = 3.0; // Adjust as needed
}
)";

const char *fragmentShaderSource = R"(
    #version 150 core
    out vec4 FragColor;
    void main() {
        FragColor = vec4(1.0, .3, .1, 1.0);
    }
)";

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void processPairs(float *positions, 
                             float *velocities, 
                             float *masses,
                             float *radii, 
                             int pointCount, int totalPairs) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < totalPairs) {
    // Recover (i, j) from idx        
    // Step 1: Find i
    int i = 0;
    int sum = 0;
    while (sum + (pointCount - 1 - i) <= idx) {
        sum += (pointCount - 1 - i);
        i++;
    }

    // Step 2: Find j
    int j = idx - sum + i + 1;
    // printf("Distance %d %d\n", i, j);

    // if (i==0) printf("%.3f %.3f %.3f    ", velocities[0], velocities[1], velocities[2]);
    
    float dx = positions[j * 3] - positions[i * 3];
    float dy = positions[j * 3 + 1] - positions[i * 3 + 1];
    float dz = positions[j * 3 + 2] - positions[i * 3 + 2];

    float distSq = dx * dx + dy * dy + dz * dz;
    if (distSq < 1e-4f)
      distSq = 1e-4f;

    float invDist = 1.0f / sqrtf(distSq);
    float invDist3 = invDist * invDist * invDist;

    // if (i==0) printf("AAAA%.3f %.3f %.3f %.3f\n", G, masses[i], masses[j], invDist3);
    float force = G * masses[i] * masses[j] * invDist3;

    // if (i==0) printf("%.3f %.3f %.3f\n", dx, force, masses[i]);
    velocities[i * 3] += dx * force / masses[i];
    velocities[i * 3 + 1] += dy * force / masses[i];
    velocities[i * 3 + 2] += dz * force / masses[i];

    velocities[j * 3] -= dx * force / masses[j];
    velocities[j * 3 + 1] -= dy * force / masses[j];
    velocities[j * 3 + 2] -= dz * force / masses[j];
    // if (i==0) printf("%.3f %.3f %.3f\n", velocities[0], velocities[1], velocities[2]);
  }
}

__global__ void processPoints(float *positions, float *velocities, int pointCount) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < pointCount) {
      positions[i * 3] += velocities[i * 3];
      positions[i * 3 + 1] += velocities[i * 3 + 1];
      positions[i * 3 + 2] += velocities[i * 3 + 2];
  }
}

int main() {
  if (!glfwInit())
    return -1;

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  GLFWwindow *window =
      glfwCreateWindow(1280, 720, "Dear ImGui Example", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Failed to initialize GLEW\n");
    glfwDestroyWindow(window);
    glfwTerminate();
    return -1;
  }

  glfwSwapInterval(1);

  setupImGui(window);

  ImFontConfig fontConfig;
  fontConfig.RasterizerDensity = 2.f;
  ImGuiIO &io = ImGui::GetIO();
  io.Fonts->AddFontDefault(&fontConfig);

  double avgFrameTime = 0;

  const int N = 3000;
  static float positions[N * 3], velocities[N * 3];
  static float masses[N], radii[N];
  static int pointCount = 0;

  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0.0, 100.0);

  // Initialize Points
  for (int i = 0; i < N; i++) {
    float curve = 2 * 3.14159 * (rand() / float(RAND_MAX));
    float radius = 1 * (rand() / float(RAND_MAX));
    float x = cos(curve) - sin(curve);
    float y = cos(curve) + sin(curve);
    positions[3 * i] = radius * x; // 1.5 * ((rand() / float(RAND_MAX)) - 0.5);
    positions[3 * i + 1] =
        radius * y; // 1.5 * ((rand() / float(RAND_MAX)) - 0.5);
    positions[3 * i + 2] = 0.02 * ((rand() / float(RAND_MAX)) - 0.5);
    float vel = sqrt(6.67e-11 * N * 200 / radius) * 0.05;

    velocities[3 * i] = -y * vel; // 0.0 * ((rand() / float(RAND_MAX)) - 0.5);
    velocities[3 * i + 1] = x * vel; // 0.0 * ((rand() / float(RAND_MAX)) -
                                     // 0.5);
    velocities[3 * i + 2] = 0.0; // 0.0 * ((rand() / float(RAND_MAX)) - 0.5);
    masses[i] = 7;
    radii[i] = 2;
  }
  pointCount = N;
  GLuint shaderProgram =
      createShaderProgram(vertexShaderSource, fragmentShaderSource);
  if (shaderProgram == 0) {
    fprintf(stderr, "Failed to create shader program\n");
    glfwDestroyWindow(window);
    glfwTerminate();
    return -1;
  }

  float *dpositions, *dvelocities, *dmasses, *dradii;
  cudaMalloc((void **)&dpositions, N * 3 * sizeof(float));
  cudaMalloc((void **)&dvelocities, N * 3 * sizeof(float));
  cudaMalloc((void **)&dmasses, N * sizeof(float));
  cudaMalloc((void **)&dradii, N * sizeof(float));

  cudaMemcpy(dpositions, positions, N*3*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dvelocities, velocities, N*3*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dmasses, masses, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dradii, radii, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  cudaCheckErrors("cudaMemcpy H2D failure");

  int totalPairs = (pointCount * (pointCount - 1)) / 2;
  int tpb = 256; // threads per block
  int bpg_pairs = (totalPairs + tpb - 1) / tpb;  // blocks per grid
  int bpg_single = (pointCount + tpb - 1) / tpb; // blocks per grid

  while (!glfwWindowShouldClose(window)) {
    double currentTime = glfwGetTime();
    glfwPollEvents();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::Begin(
        "Root", nullptr,
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoBringToFrontOnFocus |
            ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground |
            ImGuiWindowFlags_NoSavedSettings |
            ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);

    // printf("%.3f %.3f %.3f    ", velocities[0], velocities[1], velocities[2]);
    processPairs<<<bpg_pairs, tpb>>>(dpositions, dvelocities, dmasses, dradii, pointCount, totalPairs);
    cudaCheckErrors("Pair failure");
    cudaDeviceSynchronize();
    cudaCheckErrors("Sync failure Pair");
    processPoints<<<bpg_single, tpb>>>(dpositions, dvelocities, pointCount);
    cudaCheckErrors("Single failure");
    cudaDeviceSynchronize();
    cudaCheckErrors("Sync failure Single");

    cudaMemcpy(positions, dpositions, N*3 * sizeof(float), cudaMemcpyDeviceToHost);
    // printf("%.3f %.3f %.3f\n", velocities[0], velocities[1], velocities[2]);

    ImGui::End();

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions,
                 GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                          (void *)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(positions), positions);

    glBindVertexArray(vao);
    float aspectRatio = width / (float)height;
    glUniform1f(glGetUniformLocation(shaderProgram, "aspectRatio"),
                aspectRatio);
    glUseProgram(shaderProgram);

    // Enable additive blending
    glEnable(GL_BLEND);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    glDrawArrays(GL_POINTS, 0, pointCount);

    glLoadIdentity();
    glBindVertexArray(0);
    double frameTime = glfwGetTime() - currentTime;
    ImGui::Begin("Performance");
    ImGui::Text("Frame time: %.3f ms", frameTime * 1000);
    ImGui::Text("%.3f %.3f %.3f", positions[0], positions[1], positions[2]);
    ImGui::End();

    ImGui::Render();

    // Keep 60 fps
    if (frameTime < 1.0 / 60.0) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds((int)((1.0 / 60.0 - frameTime) * 1000)));
    }

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
  }

  cleanupImGui();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
