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

// #define G 6.67408e-11

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
__global__ void step(float *positions, float *velocities, float *masses,
                     int pointCount, int totalPairs) {
  __shared__ int pairs;
  pairs = 0;
  __syncthreads();
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float G = 6.67408e-11 * 1000;
  if (idx < totalPairs) {
    int i = 0;
    int sum = 0;
    while (sum + (pointCount - 1 - i) <= idx) {
        sum += (pointCount - 1 - i);
        i++;
    }

    // Step 2: Find j
    int j = idx - sum + i + 1;

    if (i!=j){
      pairs++;
      // Calcualate Interaction
      float dx = positions[j * 3] - positions[i * 3];
      float dy = positions[j * 3 + 1] - positions[i * 3 + 1];
      float dz = positions[j * 3 + 2] - positions[i * 3 + 2];

      float distSq = dx * dx + dy * dy + dz * dz;
      if (distSq < 1e-4f)
        distSq = 1e-4f;

      float invDist = 1.0f / sqrtf(distSq);
      float invDist3 = invDist * invDist * invDist;

      float force = G * masses[i] * masses[j] * invDist3;
      // if(i==0 && j == 1){
      //   printf("P1%.25f %.25f %.25f\n", positions[0], positions[1],positions[2]);
      //   printf("P2%.25f %.25f %.25f\n", positions[3], positions[4],positions[5]);
      //   printf("D%.25f\n", invDist3);
      //   printf("G%.25f\n", G);
      //   printf("F%.25f\n", force);
      // }
      velocities[i * 3] += dx * force / masses[i];
      velocities[i * 3 + 1] += dy * force / masses[i];
      velocities[i * 3 + 2] += dz * force / masses[i];

      velocities[j * 3] -= dx * force / masses[j];
      velocities[j * 3 + 1] -= dy * force / masses[j];
      velocities[j * 3 + 2] -= dz * force / masses[j];
    }
  }

  // printf("%d = %d\n", pairs, pointCount*pointCount);
  // __syncthreads();
  // if (idx < pointCount) {
  //   positions[idx * 3] += velocities[idx * 3];
  //   positions[idx * 3 + 1] += velocities[idx * 3 + 1];
  //   positions[idx * 3 + 2] += velocities[idx * 3 + 2];
  // }
}
__global__ void update(float *positions, float *velocities, int pointCount) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < pointCount) {
      positions[i * 3] += velocities[i * 3];
      positions[i * 3 + 1] += velocities[i * 3 + 1];
      positions[i * 3 + 2] += velocities[i * 3 + 2];
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

  const int N = 2e3;
  float *positions, *velocities, *masses;
  cudaMallocHost(&positions, N * 3 * sizeof(float));
  cudaMallocHost(&velocities, N * 3 * sizeof(float));
  cudaMallocHost(&masses, N * sizeof(float));

  // Initialize positions, velocities, and masses
  std::default_random_engine generator(42);
  std::normal_distribution<float> distribution(0.0, 100.0);
  for (int i = 0; i < N; i++) {
    float curve = 2 * 3.14159 * (rand() / float(RAND_MAX));
    float radius = 1 * (rand() / float(RAND_MAX));
    float x = cos(curve) - sin(curve);
    float y = cos(curve) + sin(curve);
    float vel = sqrt(6.67e-11 * N * 200 / radius) * 0.05;
    // if (i==0){
    //   printf("C%.25f\n", curve);
    //   printf("R%.25f\n", radius);
    //   printf("X%.25f\n", x);
    //   printf("Y%.25f\n", y);
    //   printf("V%.25f\n", vel);
    // }
    positions[3 * i] = radius * x;
    positions[3 * i + 1] = radius * y;
    positions[3 * i + 2] = 0.02 * ((rand() / float(RAND_MAX)) - 0.5);

    velocities[3 * i] = -y * vel;
    velocities[3 * i + 1] = x * vel;
    velocities[3 * i + 2] = 0.0;
    masses[i] = 7.;
    // radii[i] = 2;
  }
  // printf("P1%.25f %.25f %.25f\n", positions[0], positions[1],positions[2]);
  // printf("P2%.25f %.25f %.25f\n", positions[3], positions[4],positions[5]);
  int pointCount = N;

  float *d_positions, *d_velocities, *d_masses;
  cudaMalloc(&d_positions, N * 3 * sizeof(float));
  cudaMalloc(&d_velocities, N * 3 * sizeof(float));
  cudaMalloc(&d_masses, N * sizeof(float));
  cudaMemcpy(d_positions, positions, N * 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_velocities, velocities, N * 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_masses, masses, N * sizeof(float), cudaMemcpyHostToDevice);

  GLuint shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
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
  glBufferData(GL_ARRAY_BUFFER, N * 3 * sizeof(float), positions, GL_DYNAMIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  float aspectRatio = static_cast<float>(width) / static_cast<float>(height);
  
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    
    double currentTime = glfwGetTime();
    cudaMemcpy(positions, d_positions, N * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(shaderProgram);
    GLuint aspectRatioLoc = glGetUniformLocation(shaderProgram, "aspectRatio");
    glUniform1f(aspectRatioLoc, aspectRatio);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(positions), positions);
    glBufferData(GL_ARRAY_BUFFER, N * 3 * sizeof(float), positions, GL_DYNAMIC_DRAW);
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

    int blockSize = 128;
    int numBlocks = (N*N + blockSize - 1) / blockSize;
    // printf("NBlocks %d\n", numBlocks);
    int totalPairs = (N * (N - 1)) / 2;
    int numPairBlocks = (totalPairs + blockSize - 1) / blockSize;

    step<<<numPairBlocks, blockSize>>>(d_positions, d_velocities, d_masses, N, totalPairs);
    cudaDeviceSynchronize();
    cudaCheckErrors("STEP Kernel execution failed");
    update<<<numBlocks, blockSize>>>(d_positions, d_velocities, N);
    cudaDeviceSynchronize();
    cudaCheckErrors("UPDATE Kernel execution failed");
    ImGui::End();

    
    double frameTime = glfwGetTime() - currentTime;
    ImGui::SetNextWindowSize(ImVec2(200, 70));
    ImGui::SetNextWindowPos(ImVec2(10, 10));
    ImGui::Begin("Performance");
    ImGui::Text("Frame time: %.3f ms", frameTime * 1000);
    ImGui::Text("FPS: %.3f", 1000/(frameTime * 1000));
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
  cudaFree(d_masses);
  cudaFreeHost(positions);
  cudaFreeHost(velocities);
  cudaFreeHost(masses);

  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
