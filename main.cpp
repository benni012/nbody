#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cmath>
#include <random>
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/Dense"

using namespace Eigen;

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
    glfwSwapInterval(1);

    setupImGui(window);

    ImFontConfig fontConfig;
    fontConfig.RasterizerDensity = 2.f;
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontDefault(&fontConfig);

    bool initialized = false;
    double avgFrameTime = 0;

    const int N = 10000;
    static float positions[N * 3], velocities[N * 3];
    static float masses[N], radii[N];
    static int pointCount = 0;

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

        if (!initialized) {
            std::default_random_engine generator;
            std::normal_distribution<float> distribution(0.0, 100.0);

            for (int i = 0; i < N; i++) {
                positions[i * 3] = 640 + distribution(generator);
                positions[i * 3 + 1] = 360 + distribution(generator);
                positions[i * 3 + 2] = distribution(generator);

                velocities[i * 3] = distribution(generator) / 100;
                velocities[i * 3 + 1] = distribution(generator) / 100;
                velocities[i * 3 + 2] = distribution(generator) / 100;

                masses[i] = 1;
                radii[i] = 2;
            }
            pointCount = N;
            initialized = true;
        }

        for (int i = 0; i < pointCount; i++) {
            ImGui::GetWindowDrawList()->AddCircleFilled(
                    ImVec2(positions[i * 3], positions[i * 3 + 1]),
                    radii[i],
                    IM_COL32(255, 255, 0, 255)
            );
        }

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

                float force = masses[i] * masses[j] * invDist3;

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

        double frameTime = glfwGetTime() - currentTime;
        ImGui::Begin("Performance");
        ImGui::Text("Frame time: %.3f ms", frameTime * 1000);
        ImGui::End();

        ImGui::Render();
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    cleanupImGui();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
