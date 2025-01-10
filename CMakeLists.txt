cmake_minimum_required(VERSION 3.10)
project(nbody LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Check if CUDA is available
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
    enable_language(CUDA)
    set(HAS_CUDA TRUE)
else()
    message(WARNING "CUDA compiler not found. GPU support will not be enabled.")
    set(HAS_CUDA FALSE)
endif()

# Executable nbody
add_executable(nbody
    nbody_cuda.cuh
    nbody_cuda.cu
    graphics.h
    structures.h
    nbody_cpu.h
    main.cpp
)

# OMP
find_package(OpenMP)

# GL and IMGUI
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

# Link CPU version
if(OpenMP_CXX_FOUND)
    target_link_libraries(nbody OpenMP::OpenMP_CXX)
endif()

target_link_libraries(nbody
    glfw
    OpenGL::GL
    GLEW::GLEW
    imgui
)

# GPU version (CUDA-based)
if(HAS_CUDA)
    message(STATUS "Adding GPU support with CUDA.")
    # https://developer.nvidia.com/blog/cuda-pro-tip-flush-denormals-confidence/
    # https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#math-libraries
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_60 -rdc=true -ftz=true -O3")
    add_compile_definitions(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(nbody PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    message(WARNING "CUDA not found. GPU version will not be built.")
endif()

target_compile_options(nbody PRIVATE -O3)

