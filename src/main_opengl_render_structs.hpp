#include "datatypes.h"
#include "primitiveDatatypes.h"
#include <glm/fwd.hpp>
#include <memory>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#pragma once

enum class RenderState {
    pointcloud,
    spheremesh
};

struct GLWindowGlobals {
    int width  = 800;
    int height = 600;

    float lastX = width * 0.5f;
    float lastY = height * 0.5f;

    float pitch = 0.0f;
    float yaw   = -90.0f;
    float roll  = 0.0f;
    float fov   = 80.0f;

    glm::vec3 cameraPos   = {0.0f, 0.0f,  3000.0f};
    glm::vec3 cameraFront = {0.0f, 0.0f, -1.0f};
    glm::vec3 cameraUp    = {0.0f, 1.0f,  0.0f};

    float cameraSpeed = 10.0f;

    float deltaTime = 0.0f; // Time between current frame and last frame
    float lastFrame = 0.0f; // Time of last frame

    float far_plane_view_distance = 1'000'000.0f;
    float fps = 60.0f;

    float average_pixel_size = 3.0f;
    float maximum_pixel_size = 15.0;

    double average_asteroid_mass = 0.0;

    bool opengl_mouse_disabled    = true;
    bool system_analytics_window  = false;
    bool system_control           = false;

    ImVec4 clear_color = ImVec4(6.0f / 255.0f, 11.0f / 255.0f, 13.0f / 255.0f, 1.0f);

    uint8_t middle_button_state_last_frame = GLFW_RELEASE;

    float point_draw_size = 2.0f;

    bool paused = true;
    size_t paused_state_execute_frames = 0;

    size_t metric_log_length = 1;
    size_t metric_log_current_position = 0;
    std::unique_ptr<SystemMetrics[]> metric_log;
    fd_shape uv_sphere;

    bool draw_spheres = true;

    GLuint spheremesh_shader_program;

    glm::vec3 sphere_color {0.8, 0.2, 0.2};
    glm::vec3 line_color {0.2, 0.2, 0.8};

    size_t frame_count;
};

struct SimulationGLobals {
    simulation_description desc;
    integrator* inte = nullptr;
    double current_simulation_time;
    double combined_energy_current;
    double perc_energy_divergence;
};
