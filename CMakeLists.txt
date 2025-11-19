# cmake_minimum_required(VERSION 3.16)
# project(nbody VERSION 1.0 LANGUAGES CXX)
#
# # Set the C++ standard globally.
# set(CMAKE_CXX_STANDARD 23)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#
#
# # Set a default build type if none is specified.
# if(NOT CMAKE_BUILD_TYPE)
#   message(STATUS "No build type specified, defaulting to Debug")
#   set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build (Debug, Release)" FORCE)
# endif()
#
# # Common compile options for Unix-like systems.
# if(UNIX)
#   add_compile_options(-Wall -pedantic-errors)
# endif()
#
# # --------------------------------------------------------------------
# # Additional options for a Debug (testing) build.
# if(CMAKE_BUILD_TYPE MATCHES Debug)
#   message(STATUS "Configuring Debug build: enabling assertions and extra bounds checks")
#   add_compile_definitions(DEBUG_BUILD)
#   if(UNIX)
#     add_compile_options(-O3)
#   endif()
# endif()
#
# find_package(Eigen3 REQUIRED NO_MODULE)
# # --------------------------------------------------------------------
# # Set up Catch2 via FetchContent.
# # include(FetchContent)
# # FetchContent_Declare(
# #   catch2
# #   GIT_REPOSITORY https://github.com/catchorg/Catch2.git
# #   GIT_TAG v3.4.0   # use the desired version tag
# # )
# # FetchContent_MakeAvailable(catch2)
#
# # --------------------------------------------------------------------
# # Unit Testing Setup in the main CMake file.
#
# find_package(Catch2 3 REQUIRED)
# # These tests can use the Catch2-provided main
# add_executable(tests test_main.cpp)
#
# target_include_directories(tests PRIVATE
#   "${CMAKE_SOURCE_DIR}/src"
#   ${OPENGL_INCLUDE_DIR}
#   ${GLEW_INCLUDE_DIRS})
#
# target_link_libraries(tests PRIVATE 
#     Catch2::Catch2WithMain
# )
#
# # These tests need their own main
# include(CTest)
# include(Catch)
# # catch_discover_tests(unit_tests)
# # --------------------------------------------------------------------
# # Configure main executable target.
# file(GLOB SRC_FILES
#   "${CMAKE_SOURCE_DIR}/src/*.cpp")
#
# add_executable(${PROJECT_NAME} ${SRC_FILES})
# target_include_directories(${PROJECT_NAME} PRIVATE
#   "${CMAKE_SOURCE_DIR}/src"
#   ${OPENGL_INCLUDE_DIR}
#   ${GLEW_INCLUDE_DIRS})
#
# set_target_properties(${PROJECT_NAME} PROPERTIES
#   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
#
#
# target_link_libraries(${PROJECT_NAME} PRIVATE
#   GL
#   GLEW
#   glfw
#   # Eigen3::Eigen
#   Catch2WithMain
#   )
#
# # --------------------------------------------------------------------
# # Custom Target to Build and Run the Main Application.
# add_custom_target(run
#   COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${PROJECT_NAME}
#   COMMAND "${CMAKE_SOURCE_DIR}/bin/${PROJECT_NAME}"
#   WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#   COMMENT "Building and running ${PROJECT_NAME}")

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

# 4) Collect your application sources
file(GLOB_RECURSE APP_SOURCES
     ${CMAKE_SOURCE_DIR}/src/*.cpp
)
# Exclude any test-files from the main app
list(FILTER APP_SOURCES EXCLUDE REGEX ".*test.*")

# 5) Build the nbody executable
add_executable(nbody ${APP_SOURCES})
target_include_directories(nbody PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(nbody
  PRIVATE
    GL
    GLEW
    glfw
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
)

# 7) Enable CTest integration
include(CTest)
enable_testing()
add_test(NAME all_tests COMMAND NBodyTests)
