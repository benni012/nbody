#ifndef NBODY_GRAPHICS_H
#define NBODY_GRAPHICS_H
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "imgui.h"
#include "octree.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <getopt.h>
#include <random>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// exit
#include <cstdlib>

extern octree_t octree;


static GLFWwindow *window = nullptr;
static GLuint shaderProgram = 0;
static GLuint vao, vbo;
static int width = 720;
static int height = 720;
static float aspectRatio = static_cast<float>(width) / static_cast<float>(height);

// Function to compile shader
static GLuint compileShader(GLenum type, const char *source) {
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
static GLuint createShaderProgram(const char *vertexSource,
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

// shader to draw points
static const char *vertexShaderSource = R"(
    #version 330 core
    layout(location = 0) in vec3 position;    // Position attribute
    layout(location = 1) in vec4 in_color;  // Color attribute
    uniform mat4 view;
    uniform mat4 projection;
    uniform float zoom;
    out vec4 color;

    void main() {
        gl_Position = projection * view * vec4(position * zoom, 1.0);
        gl_PointSize = 3.0;
        color = in_color;
    }
)";

static const char *fragmentShaderSource = R"(
    #version 330 core
    uniform int pointCount;
    out vec4 FragColor;
    in vec4 color;
    void main() {
        FragColor = color;
    }
)";

static void setup_ImGui() {
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
    ImGuiIO &io = ImGui::GetIO();
    io.Fonts->AddFontDefault(&fontConfig);
}

static void octree_draw(octree_t *octree, int idx, int depth = 0) {
    //    return;
    const int draw_depth = 1;
    int children = octree->nodes[idx].children;

    box_t box = octree->nodes[idx].box;
    float3 center = box.center;
    float half = box.half_extent;

    // scale from -1 to 1 to 0 to width/height and also y reversed
    center.x = (center.x + 1) / 2 * width;
    center.y = height - (center.y + 1) / 2 * height;

    // imgui
    if (depth == draw_depth)
        ImGui::GetWindowDrawList()->AddRect(
                ImVec2(center.x - half / 2 * width, center.y - half / 2 * height),
                ImVec2(center.x + half / 2 * width, center.y + half / 2 * height),
                IM_COL32(0, 255, 0, 100));

    // draw point if any
    if (octree->nodes[idx].center_of_mass.w != 0 && children != ROOT) {
        float4 position = octree->nodes[idx].center_of_mass;
        position.x = (position.x + 1) / 2 * width;
        position.y = height - (position.y + 1) / 2 * height;
        //        ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(position.x,
        //        position.y), fmax(1, fmin(10, 10-depth)), IM_COL32(0, 0, 255,
        //        255));
        if (depth == draw_depth) {
            ImGui::GetWindowDrawList()->AddCircleFilled(
                    ImVec2(position.x, position.y), 5, IM_COL32(0, 0, 255, 255));
        }
    }
    if (children == ROOT) {
        return;
    }
    for (int i = 0; i < 8; i++) {
        octree_draw(octree, children + i, depth + 1);
    }
}

static void init_graphics(int N, body_t *bodies) {
    if (!glfwInit())
        exit(-1);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    window =
            glfwCreateWindow(width, height, "Dear ImGui Example", nullptr, nullptr);
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

    setup_ImGui();

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

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(body_t), (void *)0);
    glEnableVertexAttribArray(0);


    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

//    glEnable(GL_DEPTH_TEST);
    glDisable(GL_DEPTH_TEST);

}

static void draw(body_t *bodies, int N, float frameTime, float zoom, float phi, float theta) {
    float camX = 1 * sin(phi) * cos(theta);
    float camY = 1 * cos(phi); // Controls up/down movement
    float camZ = 1 * sin(phi) * sin(theta);
    
    glm::vec3 cameraPos = glm::vec3(camX, camY, camZ);
    glm::vec3 target = glm::vec3(0.0f, 0.0f, 0.0f);  // Always look at the center
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);      // Define the "up" direction

    glm::mat4 view = lookAt(cameraPos, target, up);


    glfwPollEvents();

    double currentTime = glfwGetTime();

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    // glClear(GL_COLOR_BUFFER_BIT);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shaderProgram);
    GLuint aspectRatioLoc = glGetUniformLocation(shaderProgram, "aspectRatio");
    glUniform1f(aspectRatioLoc, aspectRatio);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, N * sizeof(body_t), bodies, GL_DYNAMIC_DRAW);
    // Enable position and color attributes in body_t
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(body_t), (void*)offsetof(body_t, position)); // Assume position is vec3
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(body_t), (void*)offsetof(body_t, color)); // Assume color is vec4
    glEnableVertexAttribArray(1);


    glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspectRatio, 0.1f, 100.0f);
    GLuint projLoc = glGetUniformLocation(shaderProgram, "projection");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
    GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, value_ptr(view));
    GLuint zoomLoc = glGetUniformLocation(shaderProgram, "zoom");
    glUniform1f(zoomLoc, zoom);

    glEnable(GL_BLEND);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glDrawArrays(GL_POINTS, 0, N);
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
            ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
            ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoSavedSettings |
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

static void cleanupImGui() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

static void cleanup_graphics(body_t *bodies) {
    cleanupImGui();
    free(bodies);
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteProgram(shaderProgram);
    glfwDestroyWindow(window);
    glfwTerminate();
}

static float populate(body_t *bodies, int N, int method) {
    std::default_random_engine generator(42);
    std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
    std::normal_distribution<float> normal_dist(0.0f, 100.0f);
    std::random_device rd;
    std::mt19937 gen(rd());

    if (method == 0) {  
        // Circular disk method
        for (int i = 0; i < N; i++) {
            float angle = 2 * M_PI * (rand() / float(RAND_MAX));
            float radius = 0.5 * (rand() / float(RAND_MAX));
            float x = cos(angle) - sin(angle);
            float y = cos(angle) + sin(angle);
            float vel = sqrt(6.67e-11 * N * 200 / radius) * 0.05;
            bodies[i].position = {radius * x, radius * y, static_cast<float>(0.02 * ((rand() / float(RAND_MAX)) - 0.5)), .01f};
            bodies[i].velocity = {-y * vel, x * vel, 0.0f};
        }
        return 1;
    } else if (method == 1) {  
        // Uniform disk with central mass
        float inner_radius = 25.0f;
        float outer_radius = std::sqrt(static_cast<float>(N)) * 5.0f;
        float center_mass = 1e6;

        bodies[0].position = {0, 0, 0, center_mass};
        bodies[0].velocity = {0, 0, 0};

        for (int i = 1; i < N; ++i) {
            float angle = uniform_dist(generator) * 2.0f * M_PI;
            float sin_a = std::sin(angle);
            float cos_a = std::cos(angle);
            float t = inner_radius / outer_radius;
            float r = std::sqrt(t * t + (1.0f - t * t) * uniform_dist(generator));
            float radius = outer_radius * r;
            bodies[i].position = {radius * cos_a, radius * sin_a, 0.02f * (uniform_dist(generator) - 0.5f), 1.0f};
            float velocity_mag = std::sqrt(6.67e-11f * center_mass / radius) * 0.05f;
            bodies[i].velocity = {-sin_a * velocity_mag, cos_a * velocity_mag, 0.0f};
        }
        return 0.004 * 1/(std::sqrt(static_cast<float>(N))*0.05);
    } else if (method == 2) {  
        // Modified uniform disk
        float inner_radius = 25.0f;
        float outer_radius = std::sqrt(static_cast<float>(N)) * 5.0f;
        float center_mass = 1e6;

        bodies[0].position = {0, 0, 0, center_mass};
        bodies[0].velocity = {0, 0, 0};

        for (int i = 1; i < N; i++) {
            float angle = uniform_dist(generator) * 2.0f * M_PI;
            float sin_a = std::sin(angle);
            float cos_a = std::cos(angle);
            float t = inner_radius / outer_radius;
            float r = uniform_dist(generator) * (1.0f - t * t) + t * t;
            float radius = outer_radius * std::sqrt(r);
            bodies[i].position = {cos_a * radius, sin_a * radius, 0, 1.0f};
            bodies[i].velocity = {sin_a, -cos_a, 0};
        }

        std::sort(bodies, bodies + N, [](const body_t &a, const body_t &b) {
            return a.position.x * a.position.x + a.position.y * a.position.y <
                   b.position.x * b.position.x + b.position.y * b.position.y;
        });

        float mass = 0.0f;
        for (int i = 0; i < N; i++) {
            mass += bodies[i].position.w;
            if (bodies[i].position.x == 0 && bodies[i].position.y == 0) {
                continue;
            }
            float v = std::sqrt(mass / std::sqrt(bodies[i].position.x * bodies[i].position.x +
                                                 bodies[i].position.y * bodies[i].position.y));
            bodies[i].velocity.x *= v;
            bodies[i].velocity.y *= v;
        }
        return 0.004 * 1/(std::sqrt(static_cast<float>(N))*0.05);
    } else if (method == 3) {  
        // Plummer distribution
        for (int i = 0; i < N; ++i) {
            float x1 = uniform_dist(gen);
            float r = pow(pow(x1, -2.0f / 3.0f) - 1.0f, -0.5f);
            float x2 = uniform_dist(gen);
            float x3 = uniform_dist(gen);
            float z = r * (1 - 2 * x2);
            float x = sqrt(r * r - z * z) * cos(2 * M_PI * x3);
            float y = sqrt(r * r - z * z) * sin(2 * M_PI * x3);

            float q, x4;
            do {
                x4 = uniform_dist(gen);
                q = uniform_dist(gen);
            } while (0.1 * x4 >= q * q * pow(1 - q * q, 3.5));

            float V = q * sqrt(2) * pow(1 + r * r, -0.25);
            float x6 = uniform_dist(gen);
            float x7 = uniform_dist(gen);
            float w = (1 - 2 * x6) * V;
            float u = sqrt(V * V - w * w) * cos(2 * M_PI * x7);
            float v = sqrt(V * V - w * w) * sin(2 * M_PI * x7);

            bodies[i].position = {x, y, z, 1.0f / N};
            bodies[i].velocity = {u, v, w};
        }
        return 0.2;
    } else if (method == 4) {
        // Uniform sphere
        for (int i = 0; i < N; i++) {
            float theta = uniform_dist(generator) * 2.0f * M_PI;
            float phi = uniform_dist(generator) * M_PI;
            float r = 0.5f * std::cbrt(uniform_dist(generator));
            float x = r * std::sin(phi) * std::cos(theta);
            float y = r * std::sin(phi) * std::sin(theta);
            float z = r * std::cos(phi);
            bodies[i].position = {x, z, y, 1.0f / N};
            bodies[i].velocity = {0, 0, 0};
        }
        return 0.4;
    } else if (method == 5) {
        // central mass + uniform disk
        float center_mass = 1e6;
        bodies[0].position = {0, 0, 0, center_mass};
        bodies[0].velocity = {0, 0, 0};


        for (int i = 1; i < N; i++) {
            float angle = uniform_dist(generator) * 2.0f * M_PI;
            float r = i / 400. + 10;

            float a = r;
            float ecc = 0.4;
            float b = a * std::sqrt(1 - ecc * ecc);

            float theta = fmod(1. * i / N, 1.) * 2 * M_PI;

            // rotated ellipse
            float x = a * std::cos(angle) * std::cos(theta) - b * std::sin(angle) * std::sin(theta);
            float y = a * std::cos(angle) * std::sin(theta) + b * std::sin(angle) * std::cos(theta);

            r = std::sqrt(x * x + y * y);
            float vel = std::sqrt(center_mass * (2/ r - 1/a));

            bodies[i].position = {x, y, 0, 1.0f};
            bodies[i].velocity = {vel * std::sin(angle), -vel * std::cos(angle), 0};
            // rotate velocity by theta
            float vx = bodies[i].velocity.x * std::cos(theta) - bodies[i].velocity.y * std::sin(theta);
            float vy = bodies[i].velocity.x * std::sin(theta) + bodies[i].velocity.y * std::cos(theta);
            bodies[i].velocity.x = vx;
            bodies[i].velocity.y = vy;

            // 90% probability: set color to transparent
//            if (uniform_dist(generator) < 0.9) {
//                bodies[i].color = {0.5, 0.1, 0.05, 0.0};
//            }

        }

        return 100./N;
    }
    else {
        return -1;
    }

}
#endif // NBODY_GRAPHICS_H
