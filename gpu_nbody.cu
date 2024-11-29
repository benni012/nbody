#include <cuda.h>
#include <cuda_runtime.h>
#define IMGUI_DEFINE_MATH_OPERATORS
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

#define G 6.67408e-11
#define BLOCK_SIZE 256
#define N 7.5e3

int width = 1280;
int height = 720;

// CUDA error-checking macro
#define cudaCheckErrors(msg)                                                   \
  do {                                                                         \
    cudaError_t __err = cudaGetLastError();                                    \
    if (__err != cudaSuccess) {                                                \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg,                  \
              cudaGetErrorString(__err), __FILE__, __LINE__);                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Kernel for updating positions and velocities
__global__ void step(float4 *positions, float3 *velocities,
                     int pointCount, int totalPairs) {
  int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (particle_idx > N) return;
  float4 particle_pos = positions[particle_idx];  // load into register

  __shared__ float4 shared_pos[BLOCK_SIZE];  
  for (int i = 0; i < N; i += BLOCK_SIZE) {
    int other_idx = i + threadIdx.x;
    if (other_idx < pointCount) {
      shared_pos[threadIdx.x] = positions[other_idx];
    } else {
    shared_pos[threadIdx.x] = make_float4(0, 0, 0, 0); // Fallback for invalid threads
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < BLOCK_SIZE; j++){
      if (i+j >= N) break;
      float4 other_pos = shared_pos[j];  // load into register
      float dx = other_pos.x - particle_pos.x;
      float dy = other_pos.y - particle_pos.y;
      float dz = other_pos.z - particle_pos.z;
      float distSq = dx * dx + dy * dy + dz * dz;
      if (distSq < 1e-4f)
        distSq = 1e-4f;
      float invDist = rsqrtf(distSq);
      float invDist3 = invDist * invDist * invDist;
      float force = G * particle_pos.w * other_pos.w * invDist3;
      float fw = force / particle_pos.w; 
      velocities[particle_idx].x += dx * fw;
      velocities[particle_idx].y += dy * fw;
      velocities[particle_idx].z += dz * fw;
    }
  __syncthreads();
  }
}
__global__ void update(float4 *positions, float3 *velocities, int pointCount) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < pointCount) {
    positions[i].x += velocities[i].x;
    positions[i].y += velocities[i].y;
    positions[i].z += velocities[i].z;
  }
}

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

// Function to create shader program
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

// Shader sources
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
    FragColor = vec4(1.0, 0.3, 0.1, 1.0);
}
)";

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
      glfwCreateWindow(width, height, "nBody GPU", nullptr, nullptr);
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

  float4 *positions; 
  float3 *velocities;
  cudaMallocHost(&positions, N * sizeof(float4));
  cudaMallocHost(&velocities, N * sizeof(float3));

  // Initialize positions, velocities, and masses
  std::default_random_engine generator(42);
  std::normal_distribution<float> distribution(0.0, 100.0);
  for (int i = 0; i < N; i++) {
    float curve = 2 * 3.14159 * (rand() / float(RAND_MAX));
    float radius = 1 * (rand() / float(RAND_MAX));
    float x = cos(curve) - sin(curve);
    float y = cos(curve) + sin(curve);
    float vel = sqrt(6.67e-11 * N * 200 / radius) * 0.05;
    positions[i].w = 7.;
    positions[i].x = radius * x;
    positions[i].y = radius * y;
    positions[i].z = 0.02 * ((rand() / float(RAND_MAX)) - 0.5);

    velocities[i].x = -y * vel;
    velocities[i].y = x * vel;
    velocities[i].z = 0.0;
    // radii[i] = 2;
  }
  // int pointCount = N;

  float4 *d_positions;
  float3 *d_velocities;
  cudaMalloc(&d_positions, N * sizeof(float4));
  cudaMalloc(&d_velocities, N * sizeof(float3));
  cudaMemcpy(d_positions, positions, N * sizeof(float4), cudaMemcpyHostToDevice);
  cudaMemcpy(d_velocities, velocities, N * sizeof(float3), cudaMemcpyHostToDevice);

  GLuint shaderProgram =
      createShaderProgram(vertexShaderSource, fragmentShaderSource);
  if (shaderProgram == 0) {
    fprintf(stderr, "Failed to create shader program\n");
    glfwDestroyWindow(window);
    glfwTerminate();
    return -1;
  }
  GLuint vao, vbo;
  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);

  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, N * sizeof(float4), positions, GL_DYNAMIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  float aspectRatio = static_cast<float>(width) / static_cast<float>(height);

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    double currentTime = glfwGetTime();
    cudaMemcpy(positions, d_positions, N * sizeof(float4),
               cudaMemcpyDeviceToHost);

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(shaderProgram);
    GLuint aspectRatioLoc = glGetUniformLocation(shaderProgram, "aspectRatio");
    glUniform1f(aspectRatioLoc, aspectRatio);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, N * sizeof(float4), positions);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(float4), positions, GL_DYNAMIC_DRAW);
    glEnable(GL_BLEND);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glDrawArrays(GL_POINTS, 0, N);
    glLoadIdentity();
    glBindVertexArray(0);

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

    // glfwSwapBuffers(window);

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // printf("NBlocks %d\n", numBlocks);
    int totalPairs = (N * (N - 1)) / 2;
    int numPairBlocks = (totalPairs + BLOCK_SIZE - 1) / BLOCK_SIZE;

    step<<<numPairBlocks, BLOCK_SIZE>>>(d_positions, d_velocities, N,
                                        totalPairs);
    cudaDeviceSynchronize();
    cudaCheckErrors("STEP Kernel execution failed");
    update<<<numBlocks, BLOCK_SIZE>>>(d_positions, d_velocities, N);
    cudaDeviceSynchronize();
    cudaCheckErrors("UPDATE Kernel execution failed");
    ImGui::End();

    double frameTime = glfwGetTime() - currentTime;
    ImGui::SetNextWindowSize(ImVec2(200, 70));
    ImGui::SetNextWindowPos(ImVec2(10, 10));
    ImGui::Begin("Performance");
    ImGui::Text("Frame time: %.3f ms", frameTime * 1000);
    ImGui::Text("FPS: %.3f", 1. / frameTime);
    ImGui::End();
    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
    if (frameTime < 1.0 / 60.0) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds((int)((1.0 / 60.0 - frameTime) * 1000)));
    }
    // return 0;
  }

  cleanupImGui();
  glDeleteVertexArrays(1, &vao);
  glDeleteBuffers(1, &vbo);
  glDeleteProgram(shaderProgram);

  cudaFree(d_positions);
  cudaFree(d_velocities);
  cudaFreeHost(positions);
  cudaFreeHost(velocities);

  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
