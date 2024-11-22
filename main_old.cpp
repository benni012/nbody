#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <cstdio>
// sqrtf
#include <cmath>
// eigen
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/Dense"
// gaussian distribution
#include <random>
// fmod


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
    if (!glfwInit())
        return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // Use core profile
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // For macOS compatibility
#endif

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Dear ImGui Example", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    setupImGui(window);

    // setup hidpi font using FontConfig::RasterizationDensity
    ImFontConfig fontConfig;
    fontConfig.RasterizerDensity = 2.f;
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontDefault(&fontConfig);

    bool didput = false;
    double avgFrameTime = 0;



    while (!glfwWindowShouldClose(window)) {
        // ms
        double currentTime = glfwGetTime();
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // make new window and full screen
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
        ImGui::Begin("Root", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);

        // struct for particles, with position and velocity and mass and radius, 3d
//        typedef struct Particle {
//            Eigen::Vector3f position, velocity;
//            float mass, radius;
//        } Particle;
//
//        // array of particles
//        static Particle particles[1000];
        // struct of arrays is faster
#define N 10000
//        static Eigen::Vector3f positions[N], velocities[N];
        static float positions[N*3], velocities[N*3];
        static float masses[N], radii[N];

        // example configuration only first loop
        static int pointCount = 0;
        if (!didput) {
            std::default_random_engine generator;
            std::normal_distribution<float> distribution(0.0, 100.0);
            for (int i = 0; i < N; i++) {
////                particles[i].position = Eigen::Vector3f(rand() % 1280, rand() % 720, 0);
////                particles[i].velocity = Eigen::Vector3f(0, 0, 0);
////                particles[i].mass = 1;
////                particles[i].radius = 5;
//
////                positions[i] = Eigen::Vector3f(rand() % 1280, rand() % 720, 0);
//                // gaussian distribution around center
//
//                positions[i] = Eigen::Vector3f(640 + distribution(generator), 360 + distribution(generator), 0+ distribution(generator));
                    positions[i*3] = 640 + distribution(generator);
                    positions[i*3+1] = 360 + distribution(generator);
                    positions[i*3+2] = 0 + distribution(generator);
//                // random velocity (very small)

//                velocities[i] = Eigen::Vector3f(distribution(generator) / 100, distribution(generator) / 100, distribution(generator) / 100);
                    velocities[i*3] = distribution(generator) / 100;
                    velocities[i*3+1] = distribution(generator) / 100;
                    velocities[i*3+2] = distribution(generator) / 100;

                masses[i] = 1;
                radii[i] = 2;
            }
                // spiral
//            for (int i = 0; i < N; i++) {
//                positions[i] = Eigen::Vector3f(640 + (sin(fmod(i/1000., M_PI*2)) * 100 * i/1000.), 360 + (cos(fmod(i/1000., M_PI*2)) * 100 * i/1000.), 0);
//                velocities[i] = Eigen::Vector3f(0, 0, 0);
//                masses[i] = 1;
//                radii[i] = 2;
//            }
            // disc
            //initialize the position and velocity vectors of every particle.
//            for (int i = 0; i < N*3; i+=3) {
//                float curve = 2 * 3.14159 * (rand() / float(RAND_MAX));
//                float radius = 100 * (rand() / float(RAND_MAX));
//                float x = (cos(curve) - sin(curve));
//                float y = (cos(curve) + sin(curve));
////                vertices[i] = radius * x;//1.5 * ((rand() / float(RAND_MAX)) - 0.5);
////                vertices[i + 1] = radius * y;//1.5 * ((rand() / float(RAND_MAX)) - 0.5);
////                vertices[i+2] =  0.02 * ((rand() / float(RAND_MAX)) - 0.5);
////                positions[i] = Eigen::Vector3f(640+radius*x, 360+radius*y, 0.02 * ((rand() / float(RAND_MAX)) - 0.5));
//                float vel = sqrt(6.67e-11 * N * 200 / radius) * 0.05;
//
////                velocities[i] = -y * vel;//0.0 * ((rand() / float(RAND_MAX)) - 0.5);
////                velocities[i + 1] = x * vel;//0.0 * ((rand() / float(RAND_MAX)) - 0.5);
////                velocities[i + 2] = 0.0;//0.0 * ((rand() / float(RAND_MAX)) - 0.5);
//                velocities[i] = Eigen::Vector3f(-y * vel, x * vel, 0);
//                masses[i] = 1;
//                radii[i] = 2;
//            }
//            for (int i = 0; i < N; i++) {
//                float angle = fmod(i/1000., M_PI*2);
//                positions[i] = Eigen::Vector3f(640 + (sin(angle) * 100), 360 + (cos(angle) * 100), 0);
//                velocities[i] = Eigen::Vector3f(0, 0, 0);
//                masses[i] = 1;
//                radii[i] = 2;
//            }

            pointCount = N;
            // 2 body equilibrium
//            particles[0].position = Eigen::Vector3f(640-250, 360, 0);
//            particles[0].velocity = Eigen::Vector3f(0, 5 , 0);
//            particles[0].mass = 30000;
//            particles[0].radius = 5;
//
//            particles[1].position = Eigen::Vector3f(640+250, 360, 0);
//            particles[1].velocity = Eigen::Vector3f(0, -5, 0);
//            particles[1].mass = 30000;
//            particles[1].radius = 5;
//            pointCount = 2;

            didput = true;
        }

        // vectorized implementation
        // make the points attract each other



//        if (ImGui::IsMouseReleased(0)) {
//            // new point at mouse position
//            ImVec2 mousePos = ImGui::GetMousePos();
//            particles[pointCount].position = Eigen::Vector3f(mousePos.x, mousePos.y, 0);
//            particles[pointCount].velocity = Eigen::Vector3f(0, 0, 0);
//            particles[pointCount].mass = 1;
//            particles[pointCount].radius = 5;
//
//            pointCount = (pointCount + 1) % 1000;
//        }
        for (int i = 0; i < pointCount; i++) {
//            ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(p.position[0], p.position[1]), p.radius, IM_COL32(255, 255, 0, 255));
//            ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(positions[i][0], positions[i][1]), radii[i], IM_COL32(255, 255, 0, 255));
            // color based on z
//            ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(positions[i][0], positions[i][1]), radii[i], IM_COL32(255, 255, 0, 255));
            ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(positions[i*3], positions[i*3+1]), radii[i], IM_COL32(255, 255, 0, 255));
        }

        // make the points attract each other
//#pragma omp parallel for
//        for (int i = 0; i < pointCount; i++) {
//            for (int j = i + 1; j < pointCount; j += 1) { // Process 4 particles at a time
//                Eigen::Vector3f delta = positions[j] - positions[i];
//
//                // Compute squared distance (SIMD friendly)
//                float distSq = delta.squaredNorm();
//                if (distSq < 1e-4f) distSq = 1e-4f; // Avoid divide by zero
//
//                // Gravitational force (vectorized)
//                float invDist = 1.0f / sqrtf(distSq); // Approximate sqrt with SIMD libraries if necessary
//                float invDist3 = invDist * invDist * invDist;
//
//                Eigen::Vector3f force = delta * (masses[i] * masses[j] * invDist3);
//
//                velocities[i] += force / masses[i];
//                velocities[j] -= force / masses[j];
//            }
//        }


        // no eigen: vectorized implementation simd
//        #pragma omp parallel for
//        for (int i = 0; i < pointCount; i++) {
//            for (int j = i + 1; j < pointCount; j += 1) { // Process 4 particles at a time
//                float dx = positions[j][0] - positions[i][0];
//                float dy = positions[j][1] - positions[i][1];
//                float dz = positions[j][2] - positions[i][2];
//
//                // Compute squared distance (SIMD friendly)
//                float distSq = dx * dx + dy * dy + dz * dz;
//                if (distSq < 1e-4f) distSq = 1e-4f; // Avoid divide by zero
//
//                // Gravitational force (vectorized)
//                float invDist = 1.0f / sqrtf(distSq); // Approximate sqrt with SIMD libraries if necessary
//                float invDist3 = invDist * invDist * invDist;
//
//                float force = masses[i] * masses[j] * invDist3;
//
//                velocities[i][0] += dx * force / masses[i];
//                velocities[i][1] += dy * force / masses[i];
//                velocities[i][2] += dz * force / masses[i];
//
//                velocities[j][0] -= dx * force / masses[j];
//                velocities[j][1] -= dy * force / masses[j];
//                velocities[j][2] -= dz * force / masses[j];
//            }
//        }

        // no eigen: vectorized implementation simd
        #pragma omp parallel for
        for (int i = 0; i < pointCount; i++) {
            for (int j = i + 1; j < pointCount; j += 1) { // Process 4 particles at a time
                float dx = positions[j*3] - positions[i*3];
                float dy = positions[j*3+1] - positions[i*3+1];
                float dz = positions[j*3+2] - positions[i*3+2];

                // Compute squared distance (SIMD friendly)
                float distSq = dx * dx + dy * dy + dz * dz;
                if (distSq < 1e-4f) distSq = 1e-4f; // Avoid divide by zero

                // Gravitational force (vectorized)
                float invDist = 1.0f / sqrtf(distSq); // Approximate sqrt with SIMD libraries if necessary
                float invDist3 = invDist * invDist * invDist;

                float force = masses[i] * masses[j] * invDist3;

                velocities[i*3] += dx * force / masses[i];
                velocities[i*3+1] += dy * force / masses[i];
                velocities[i*3+2] += dz * force / masses[i];

                velocities[j*3] -= dx * force / masses[j];
                velocities[j*3+1] -= dy * force / masses[j];
                velocities[j*3+2] -= dz * force / masses[j];
            }
        }

        // damping
//        for (int i = 0; i < pointCount; i++) {
//            velocities[i] *= 0.99;
//        }

        // no eigen
        for (int i = 0; i < pointCount; i++) {
//            positions[i][0] += velocities[i][0];
//            positions[i][1] += velocities[i][1];
//            positions[i][2] += velocities[i][2];
            positions[i*3] += velocities[i*3];
            positions[i*3+1] += velocities[i*3+1];
            positions[i*3+2] += velocities[i*3+2];
        }

        ImGui::End();

        double frameTime = glfwGetTime() - currentTime;
//        avgFrameTime = avgFrameTime * 0.999 + frameTime * 0.001;
        ImGui::Begin("Performance");
        ImGui::Text("Frame time: %.3f ms", frameTime * 1000);
//        ImGui::Text("Average frame time: %.3f ms", avgFrameTime * 1000);
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
