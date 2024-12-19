//
// Created by Ben Chwalek on 09.12.24.
//

#ifndef NBODY_GRAPHICS_H
#define NBODY_GRAPHICS_H
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cmath>
#include <random>
#include <chrono>
#include <getopt.h>
#include "imgui.h"
#include "octree.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
// exit
#include <cstdlib>

extern octree_t octree;

void cleanupImGui() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}


GLFWwindow *window = nullptr;
GLuint shaderProgram = 0;
GLuint vao, vbo;
int width = 720;
int height = 720;
float aspectRatio = static_cast<float>(width) / static_cast<float>(height);

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

// shader to draw points
const char* vertexShaderSource = R"(
#version 150 core
uniform float aspectRatio;
in vec3 position;

void main() {
    vec3 adjustedPosition = position;
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

void setupImGui() {
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

    // setup hidpi font using FontConfig::RasterizationDensity
    ImFontConfig fontConfig;
    fontConfig.RasterizerDensity = 2.f;
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontDefault(&fontConfig);
}

void octree_draw(octree_t *octree, int idx, int depth = 0) {
    int children = octree->nodes[idx].children;

    box_t box = octree->nodes[idx].box;
    float3 center = box.center;
    float half = box.half_extent;

    // scale from -1 to 1 to 0 to width/height and also y reversed
    center.x = (center.x + 1) / 2 * width;
    center.y = height - (center.y + 1) / 2 * height;

    // imgui
//        ImGui::GetWindowDrawList()->AddRect(ImVec2(center.x - half/2*width, center.y - half/2*height),
//                                            ImVec2(center.x + half/2*width, center.y + half/2*height),
//                                            IM_COL32(0, 255, 0, 100));
    // draw point if any
    if (octree->nodes[idx].center_of_mass.w != 0 && children != ROOT) {
        float4 position = octree->nodes[idx].center_of_mass;
        position.x = (position.x + 1) / 2 * width;
        position.y = height - (position.y + 1) / 2 * height;
//        ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(position.x, position.y), fmax(1, fmin(10, 10-depth)), IM_COL32(0, 0, 255, 255));
        if (depth == 0) {
            ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(position.x, position.y), 5, IM_COL32(0, 0, 255, 255));
        }
    }
    if (children == ROOT) {
        return;
    }
    for (int i = 0; i < 8; i++) {
        octree_draw(octree, children + i, depth + 1);
    }
}

void initGraphics(int N, body_t *bodies) {
    if (!glfwInit()) exit(-1);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    window = glfwCreateWindow(width, height, "Dear ImGui Example", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        exit(-1);
    }
    glfwMakeContextCurrent(window);
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        exit(-1);
    }

    glfwSwapInterval(1);

    setupImGui();

    shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
    if (shaderProgram == 0) {
        fprintf(stderr, "Failed to create shader program\n");
        glfwDestroyWindow(window);
        glfwTerminate();
        exit(-1);
    }
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(body_t), bodies, GL_DYNAMIC_DRAW);


    // TODO
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(body_t), (void *)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}


void draw(body_t *bodies, int N, float frameTime) {
    glfwPollEvents();

    double currentTime = glfwGetTime();

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(shaderProgram);
    GLuint aspectRatioLoc = glGetUniformLocation(shaderProgram, "aspectRatio");
    glUniform1f(aspectRatioLoc, aspectRatio);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, N * sizeof(body_t), bodies);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(body_t), bodies, GL_DYNAMIC_DRAW);
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

//    octree_draw(&octree, ROOT);


    ImGui::End();

    ImGui::SetNextWindowSize(ImVec2(200, 70));
    ImGui::SetNextWindowPos(ImVec2(10, 10));
    ImGui::Begin("Performance");
    ImGui::Text("Frame time: %.3f ms", frameTime * 1000);
    ImGui::Text("FPS: %.3f", 1. / frameTime);
    ImGui::End();
    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
}

void cleanup(body_t *bodies) {
    cleanupImGui();
    free(bodies);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(shaderProgram);
    glfwDestroyWindow(window);
    glfwTerminate();
}

void populate(body_t *bodies, int N) {
    // Initialize positions, velocities, and masses
    std::default_random_engine generator(42);
    std::normal_distribution<float> distribution(0.0, 100.0);
    for (int i = 0; i < N; i++) {
        float curve = 2 * 3.14159 * (rand() / float(RAND_MAX));
        float radius = .5 * (rand() / float(RAND_MAX));
        float x = cos(curve) - sin(curve);
        float y = cos(curve) + sin(curve);
        float vel = sqrt(6.67e-11 * N * 200 / radius) * 0.05;
        bodies[i].position.w = 7.;
        bodies[i].position.x = radius * x;
        bodies[i].position.y = radius * y;
        bodies[i].position.z = 0.02 * ((rand() / float(RAND_MAX)) - 0.5);
//        positions[i].z = 0.0;

        bodies[i].velocity.x = -y * vel;
        bodies[i].velocity.y = x * vel;
        bodies[i].velocity.z = 0.0;
    }
}

#endif //NBODY_GRAPHICS_H
