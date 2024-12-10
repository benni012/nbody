cmake_minimum_required(VERSION 3.10)
project(nbody LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# imgui
#add_library(imgui STATIC
#    imgui/imgui.cpp
#    imgui/imgui_draw.cpp
#    imgui/imgui_widgets.cpp
#    imgui/imgui_impl_glfw.cpp
#    imgui/imgui_impl_opengl3.cpp
#)

# exectuable nbody
add_executable(nbody
        main.cpp
        nbody_cuda.h
        graphics.h
        nbody_cpu.h
)
# omp

find_package(OpenMP)
find_package(glfw3 3.3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS} imgui/)

# ImGui library
add_library(imgui STATIC
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_widgets.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_impl_glfw.cpp
    imgui/imgui_impl_opengl3.cpp
)
target_link_libraries(imgui glfw)

# nbody (CPU version) executable
add_executable(nbody main.cpp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(nbody OpenMP::OpenMP_CXX)
endif()
target_link_libraries(nbody
    glfw
    OpenGL::GL
    GLEW::GLEW
    imgui
)
target_compile_options(nbody PRIVATE -O3)

# GPU version (CUDA-based) executable
find_package(CUDA)
if(CUDA_FOUND)
    message(STATUS "CUDA found. Adding GPU support.")
    # fmad https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-for-steering-gpu-code-generation
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_60 -fmad=true")

    add_executable(gpu_nbody gpu_nbody.cu)
    target_include_directories(gpu_nbody PRIVATE imgui/ ${GLEW_INCLUDE_DIRS})
    target_link_libraries(gpu_nbody
        glfw
        OpenGL::GL
        GLEW::GLEW
        imgui
    )
    set_target_properties(gpu_nbody PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_options(gpu_nbody PRIVATE -O3)
else()
    message(WARNING "CUDA not found. GPU version will not be built.")
endif()
