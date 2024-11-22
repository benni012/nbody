# cmake for imgui, glfw
# imgui is in imgui folder

cmake_minimum_required(VERSION 3.10)
project(nbody)

# set c++ standard
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
        )
target_include_directories(nbody PRIVATE imgui/ )
find_package(glfw3 3.3 REQUIRED)
target_link_libraries(nbody glfw)
find_package(OpenGL REQUIRED)
target_link_libraries(nbody OpenGL::GL)
add_library(imgui STATIC
        imgui/imgui.cpp
        imgui/imgui_draw.cpp
        imgui/imgui_widgets.cpp
        imgui/imgui_tables.cpp
        imgui/imgui_impl_glfw.cpp
        imgui/imgui_impl_opengl3.cpp
        )
target_link_libraries(imgui glfw)
target_link_libraries(nbody imgui)