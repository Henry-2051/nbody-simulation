cmake_minimum_required(VERSION 3.14)
project(NBody VERSION 1.0 LANGUAGES CXX)

# 1) C++ settings
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 2) Put executables into bin/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

find_package(OpenGL     REQUIRED)   # provides OpenGL::GL
find_package(GLEW       REQUIRED)   # provides GLE::GLEW
find_package(glfw3 3.3  REQUIRED)   # provides glfw::glfw
# find_package(Eigen3 3.3 REQUIRED NO_MODULE)  # provides Eigen3::Eigen
find_package(Catch2 3    REQUIRED)   # provides Catch2::Catch2WithMain
find_package(imgui  REQUIRED)

# set(IMGUI_SRC_DIR "${CMAKE_SOURCE_DIR}/src/imgui")
#
# add_library(imgui_s STATIC
#     ${IMGUI_SRC_DIR}/imgui.cpp
#     ${IMGUI_SRC_DIR}/imgui_draw.cpp
#     ${IMGUI_SRC_DIR}/imgui_tables.cpp
#     ${IMGUI_SRC_DIR}/imgui_widgets.cpp
#     ${IMGUI_SRC_DIR}/backends/imgui_impl_glfw.cpp
#     ${IMGUI_SRC_DIR}/backends/imgui_impl_opengl3.cpp
# )
#
# target_include_directories(imgui_s PUBLIC
#     ${IMGUI_SRC_DIR} 
#     ${IMGUI_SRC_DIR}/backends
# )
#
# target_link_libraries(imgui_s PUBLIC
#     GL 
#     GLEW
#     glfw
# )

# 4) Collect your application sources
file(GLOB_RECURSE APP_SOURCES
     ${CMAKE_SOURCE_DIR}/src/*.cpp
)
# Exclude any test-files from the main app
list(FILTER APP_SOURCES EXCLUDE REGEX ".*test.*")

# 5) Build the nbody executable
add_executable(nbody ${APP_SOURCES})
target_include_directories(nbody PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(nbody
  PRIVATE
    GL
    GLEW
    glfw
    imgui
)

# 6) Build the Catch2 test runner
file(GLOB_RECURSE TEST_SOURCES
     ${CMAKE_SOURCE_DIR}/src/*test*.cpp
)
add_executable(NBodyTests ${TEST_SOURCES})
target_include_directories(NBodyTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(NBodyTests
  PRIVATE
    Catch2::Catch2WithMain
    GL       # if your tests need GL/GLEW as well, otherwise drop these
    GLEW
    glfw
    imgui
)

# 7) Enable CTest integration
include(CTest)
enable_testing()
add_test(NAME all_tests COMMAND NBodyTests)
