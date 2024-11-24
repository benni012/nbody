#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cmath>
#include <random>
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/Dense"
#include <chrono>
#include <thread>

int width = 1280;
int height = 720;

// Function to compile shader
GLuint compileShader(GLenum type, const char* source) {
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
GLuint createShaderProgram(const char* vertexSource, const char* fragmentSource) {
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
void setupImGui(GLFWwindow* window) {
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

// shader to draw points
const char* vertexShaderSource = R"(
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

const char* fragmentShaderSource = R"(
    #version 150 core
    out vec4 FragColor;
    void main() {
        FragColor = vec4(1.0, .3, .1, 1.0);
    }
)";


int main() {
    if (!glfwInit()) return -1;


    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Dear ImGui Example", nullptr, nullptr);
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
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontDefault(&fontConfig);

    double avgFrameTime = 0;


    const int N = 5000;
    static float positions[N * 3], velocities[N * 3];
    static float masses[N], radii[N];
    static int pointCount = 0;
    static float G = 6.67408e-11;

    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 100.0);

    for (int i = 0; i < N; i++) {
        float curve = 2 * 3.14159 * (rand() / float(RAND_MAX));
        float radius = 1 * (rand() / float(RAND_MAX));
        float x = cos(curve)-sin(curve);
        float y = cos(curve)+sin(curve);
        positions[3*i] = radius * x;//1.5 * ((rand() / float(RAND_MAX)) - 0.5);
        positions[3*i + 1] = radius * y;//1.5 * ((rand() / float(RAND_MAX)) - 0.5);
        positions[3*i+2] =  0.02 * ((rand() / float(RAND_MAX)) - 0.5);
        float vel = sqrt(6.67e-11 * N * 200 / radius) * 0.05 ;

        velocities[3*i] = -y * vel;//0.0 * ((rand() / float(RAND_MAX)) - 0.5);
        velocities[3*i + 1] = x * vel;//0.0 * ((rand() / float(RAND_MAX)) - 0.5);
        velocities[3*i + 2] = 0.0;//0.0 * ((rand() / float(RAND_MAX)) - 0.5);
        masses[i] = 7;
        radii[i] = 2;
    }
    pointCount = N;
    GLuint shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    if (shaderProgram == 0) {
        fprintf(stderr, "Failed to create shader program\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }


    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("Root", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
                     ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground |
                     ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                     ImGuiWindowFlags_NoNav);




#pragma omp parallel for
        for (int i = 0; i < pointCount; i++) {
            for (int j = i + 1; j < pointCount; j++) {
                float dx = positions[j * 3] - positions[i * 3];
                float dy = positions[j * 3 + 1] - positions[i * 3 + 1];
                float dz = positions[j * 3 + 2] - positions[i * 3 + 2];

                float distSq = dx * dx + dy * dy + dz * dz;
                if (distSq < 1e-4f) distSq = 1e-4f;

                float invDist = 1.0f / sqrtf(distSq);
                float invDist3 = invDist * invDist * invDist;

                float force = G * masses[i] * masses[j] * invDist3;

                velocities[i * 3] += dx * force / masses[i];
                velocities[i * 3 + 1] += dy * force / masses[i];
                velocities[i * 3 + 2] += dz * force / masses[i];

                velocities[j * 3] -= dx * force / masses[j];
                velocities[j * 3 + 1] -= dy * force / masses[j];
                velocities[j * 3 + 2] -= dz * force / masses[j];
            }
        }

        for (int i = 0; i < pointCount; i++) {
            positions[i * 3] += velocities[i * 3];
            positions[i * 3 + 1] += velocities[i * 3 + 1];
            positions[i * 3 + 2] += velocities[i * 3 + 2];
        }

        ImGui::End();

        GLuint vao, vbo;
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);

        glBindVertexArray(vao);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_DYNAMIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(positions), positions);

        glBindVertexArray(vao);
        float aspectRatio = width / (float)height;
        glUniform1f(glGetUniformLocation(shaderProgram, "aspectRatio"), aspectRatio);
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
        ImGui::End();

        ImGui::Render();

        if (frameTime < 1.0 / 60.0) {
            std::this_thread::sleep_for(std::chrono::milliseconds((int) ((1.0 / 60.0 - frameTime) * 1000)));
        }


        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    cleanupImGui();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
