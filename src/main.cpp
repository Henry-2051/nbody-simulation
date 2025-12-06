#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <memory>
#include <ostream>
#include <ratio>
#include <sstream>

#include <functional>
#include <glm/common.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/quaternion_transform.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/trigonometric.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <print>
#include <vector>
#include <iostream>
#include <variant>
#include <chrono>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "datatypes.h"


#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "analytics.h"

#include "simulation_generator_functions.hpp"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

bool display_opengl_shader_compilation_error(unsigned int vertexShader); 
bool display_opengl_program_compilation_error(unsigned int program);




namespace gl_window_globals {
    int width = 800;
    int height = 600;

    float lastX = float(static_cast<float>(width) / 2), lastY = float(static_cast<float>(height) / 2);

    float pitch = 0.0f;
    float yaw = -90.0f;
    float roll = 0.0f;
    float fov = 80;

    glm::vec3 cameraPos   = glm::vec3(0.0f, 0.0f,  3000.0f);
    glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f,  0.0f);

    float cameraSpeed = 10.0;

    float deltaTime = 0.0f;	// Time between current frame and last frame
    float lastFrame = 0.0f; // Time of last frame
    
    float far_plane_view_distance = 1000000.0;

    float fps = 60.0;
    float average_pixel_size = 3.0;
    float maximum_pixel_size = 15.0;
    double average_asteroid_mass;

    bool opengl_mouse_disabled {true};

    bool system_analytics_window {false};
    bool system_control {false};
    ImVec4 clear_color = ImVec4(6.0 / 255.0, 11.0 / 255.0, 13.0 / 255.0, 1.00f);

    uint8_t middle_button_state_last_frame {GLFW_RELEASE};
    float point_draw_size = 2;

    bool paused = true;
    size_t paused_state_execute_frames = 0;
    
    size_t metric_log_length = 1;
    size_t metric_log_current_position = 0;
    std::unique_ptr<SystemMetrics[]> metric_log;
}

namespace simulation_globals {
    simulation_description desc;
    integrator* inte = nullptr;
}



void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (yoffset > 0) {
        gl_window_globals::cameraSpeed *= 1.2f;
    } else {
        gl_window_globals::cameraSpeed /= 1.2f;
    }
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    bool mouse_captured = glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED;
    if (!mouse_captured) { return; }
    float xoffset = xpos - gl_window_globals::lastX;
    float yoffset = gl_window_globals::lastY - ypos; // reversed since y-coordinates range from bottom to top
    gl_window_globals::lastX = xpos;
    gl_window_globals::lastY = ypos;

    const float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    gl_window_globals::yaw   += xoffset;
    gl_window_globals::pitch += yoffset;  

    if (gl_window_globals::pitch > 89.0f) { gl_window_globals::pitch =  89.0f; }

    if (gl_window_globals::pitch < -89.0f) { gl_window_globals::pitch = -89.0f;}
};

void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    const float cameraSpeed = gl_window_globals::cameraSpeed * gl_window_globals::deltaTime; // adjust accordingly
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        gl_window_globals::cameraPos += cameraSpeed * gl_window_globals::cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        gl_window_globals::cameraPos -= cameraSpeed * gl_window_globals::cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        gl_window_globals::cameraPos -= glm::normalize(glm::cross(gl_window_globals::cameraFront, gl_window_globals::cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        gl_window_globals::cameraPos += glm::normalize(glm::cross(gl_window_globals::cameraFront, gl_window_globals::cameraUp)) * cameraSpeed;

    uint8_t middle_button_state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE);

    if (middle_button_state == GLFW_PRESS && gl_window_globals::middle_button_state_last_frame == GLFW_RELEASE) {

        if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) 
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        else if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_NORMAL) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);}

        gl_window_globals::middle_button_state_last_frame = true;
    } 
    gl_window_globals::middle_button_state_last_frame = middle_button_state;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    gl_window_globals::width = width;
    gl_window_globals::height= height;
    glViewport(0, 0, width, height);
}


double find_average_mass_of_bodies(std::vector<gravitationalBody>& bodies){
    size_t N = bodies.size();
    double total_mass {0};

    for (auto& body : bodies) {
        total_mass += body.mass;
    }

    return (total_mass / N);
}



int openglDisplay(simulation_description sim_desc) {
    unsigned int VBO, VAO, EBO, vertex_shader, fragment_shader, shader_program;
    namespace g = gl_window_globals;
    namespace gs = simulation_globals;

    gs::desc = sim_desc;

    const char * vertex_shader_source = R"glsl(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        // layout (location = 1) in vec3 aColor;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        out vec3 vertColor;

        void main()
        {
            gl_Position = projection * view * model * vec4(aPos, 1.0f);
            vertColor = vec3(1.0,1.0,1.0);
        }
    )glsl";


    const char * fragment_shader_source = R"glsl(
        #version 330 core
        out vec4 FragColor;

        in vec3 vertColor;

        void main()
        {
            FragColor = vec4(vertColor, 1.0f);
        }
    )glsl";

    std::vector<gravitationalBody> bodies = sim_desc.gen_bodies();
    // std::vector<gravitationalBody> bodies = generate_thousand_random_bodies();

    std::vector<glm::vec3> float_body_positions (bodies.size());

    // helper function to ensure float_bodies_positions mirrors the positions 
    // within bodies but using vec3 rather than dvec3 for opengl
    auto recalc_float_pos = [&bodies, &float_body_positions](){
        if (bodies.size() != float_body_positions.size()) {
            float_body_positions.resize(bodies.size());
        }
        for (int i = 0; i < bodies.size(); i ++) {
            float_body_positions[i] = bodies[i].position;
        }
    };

    double max_dimension = get_max_dimension(bodies);
    float draw_scale_factor = 1000.0 / max_dimension;
    
    // std::println("scale factor calculated as : {}", draw_scale_factor);

    auto scale_float_positions = [&float_body_positions, draw_scale_factor]() {
        for (auto& pos : float_body_positions) {
            pos *= draw_scale_factor;
        }
    };

    recalc_float_pos();
    scale_float_positions();

    double combined_energy_last = calculate_gpe(bodies) + calculate_kinetic_energy(bodies);
    double combined_energy_current;
    double perc_energy_divergence;

    g::metric_log = std::make_unique<SystemMetrics[]>(g::metric_log_length);
    g::metric_log[0] = SystemMetrics(combined_energy_last, 0);

    // now atomic 
    auto calculate_system_metrics = [&bodies, &combined_energy_current, &perc_energy_divergence, &combined_energy_last]() {
            combined_energy_current = calculate_kinetic_energy(bodies) + calculate_gpe(bodies);
            perc_energy_divergence = 100.0 * (combined_energy_current - combined_energy_last) / combined_energy_last;
    };

    // for (auto& body : bodies) {
    //     std::println("Body radius : {}", body.radius);
    //     std::println("Body mass : {}", body.mass);
    // }

    // size_t max_points = bodies.size() * 2;

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(gl_window_globals::width, gl_window_globals::height, "nbody", NULL, NULL);

    if (window == nullptr) {
        std::cerr << "Error, unable to initialise glfw window :(";
    }

    glfwMakeContextCurrent(window);
        
    glEnable(GL_DEPTH_TEST);  
    glewInit();

    glViewport(0, 0, 800, 600); // this is the size of the rendering window, 
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);


    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);  
    glfwSetScrollCallback(window, scroll_callback); 
    glfwSetCursorPosCallback(window, mouse_callback);  

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    // bind the vertex array object first, then bind and se the vertex buffer(s), and then configure vertex attributes
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // function designed to copy user defined data into the currently bound buffer
    glBufferData(GL_ARRAY_BUFFER, bodies.size() * sizeof(glm::vec3), nullptr, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // atribute pointers and data layout configuration
    // param 1 : location of the configured attibute (look at vertex shader)
    // param 2 : size of the vertex attribute, composed of 3 values
    // param 3 : the type of the data, in opengl a vec3 is composed of floats
    // param 4 : this is whether we want the data normalised, this would be true for integer data types
    // param 5 : this is known as the stride length, it tells us the length between consecutive values
    // param 6 : type (void *) tells us the offset of where the position data begins in the buffer.

    vertex_shader = glCreateShader(GL_VERTEX_SHADER);

    glShaderSource(vertex_shader, 1, &vertex_shader_source, NULL);
    glCompileShader(vertex_shader);

    display_opengl_shader_compilation_error(vertex_shader);

    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_source, NULL);

    shader_program = glCreateProgram();
    glAttachShader(shader_program, vertex_shader);
    glAttachShader(shader_program, fragment_shader);
    glLinkProgram(shader_program);

    display_opengl_program_compilation_error(shader_program);

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);  

    glm::mat4 projection, view, model;
    // this is the only place where we are conducting pointer shenanigans with this variable, its 
    // good practice to free the resource before assigning a new resource, preventing a memory leak
    
    delete gs::inte;
    gs::inte = new integrator(sim_desc.integrator, sim_desc.acceleration_function, sim_desc.collision_function, sim_desc.integrator_step_size_hint);
    
    // auto last_frame = std::chrono::high_resolution_clock::now();
    double current_simulation_time = sim_desc.start;
    double next_target_time = sim_desc.start + sim_desc.simulation_step_size;

    float time_accumulator = 0;
    size_t frame_count = 0;

    // IMGUI SETUP
    // Create window with graphics context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    ImGui::StyleColorsDark();


    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    while((!glfwWindowShouldClose(window)) &&
        current_simulation_time < sim_desc.end) 
    {
        if (!g::paused) frame_count ++;

        float current_time = glfwGetTime();
        gl_window_globals::deltaTime = current_time - gl_window_globals::lastFrame;
        gl_window_globals::lastFrame = current_time;
        time_accumulator += gl_window_globals::deltaTime;

        // calculate_gpe is a relatively expensive call of order n^2, comparable to timestepping the system
        if ((!g::paused) && g::system_analytics_window) {
            calculate_system_metrics();
        }

        // IMGUI DISPLAY
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        {
            ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.
            //
            // ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
            ImGui::Checkbox("System analytics", &g::system_analytics_window);      // Edit bools storing our window open/close state
            ImGui::Checkbox("System control", &g::system_control);

            ImGui::Checkbox("paused", &g::paused);
            ImGui::ColorEdit3("clear color", (float*)&gl_window_globals::clear_color); // Edit 3 floats representing a color
            ImGui::SliderFloat("point size", &g::point_draw_size, 1.0f, 20.0f);


            ImGui::Text("Simulation world time: %.0f seconds\nSimulation progress: %.1f%%", current_simulation_time, (((current_simulation_time - sim_desc.start) * 100.0) / (sim_desc.end - sim_desc.start)));
            ImGui::Text("Simulation frame: %zu", frame_count);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }

        if (g::system_analytics_window) {
            ImGui::Begin("System analtics", &g::system_analytics_window);

            ImGui::Text("%s", std::format("current energy of the systme  : {:.0f} Joules", combined_energy_current).c_str());
            ImGui::Text("%s", std::format("percentage energy change from last frame : {:.2e}%%", perc_energy_divergence).c_str());
            double perc_energy_change_from_start = 100.0 * ((combined_energy_current - g::metric_log[0].combined_energy) / g::metric_log[0].combined_energy);
            ImGui::Text("%s", std::format("percentage energy change from start : {:.2e}%%", perc_energy_change_from_start).c_str());
            ImGui::End();
        }

        if (g::system_control) {
            ImGui::Begin("System control", &g::system_control);
            double min = sim_desc.simulation_step_size * 0.1, max = sim_desc.simulation_step_size * 10.0;
            ImGui::SliderScalar("simulation time per frame", ImGuiDataType_Double, 
                                &gs::desc.simulation_step_size, &min, &max);

            min = sim_desc.integrator_step_size_hint* 0.1, max = sim_desc.integrator_step_size_hint * 10.0;
            ImGui::SliderScalar("integrator step size", ImGuiDataType_Double, &gs::inte->step_size, &min, &max);
            ImGui::End();
        }

        if (!g::paused) {
            combined_energy_last = combined_energy_current;

            current_simulation_time = gs::inte->integrate(bodies, current_simulation_time, next_target_time);
            next_target_time += gs::desc.simulation_step_size;
            recalc_float_pos();
            scale_float_positions();
        }

        // for (auto& pos : float_body_positions) {
        //     std::println("({}, {}, {})", pos.x, pos.y, pos.z);
        // }

        // just_print_glm_vec3(gl_window_globals::cameraPos);
        // std::println();

        processInput(window);
        glm::vec3 direction= glm::vec3(
            cos(glm::radians(gl_window_globals::yaw)) * cos(glm::radians(gl_window_globals::pitch)),
            sin(glm::radians(gl_window_globals::pitch)),
            sin(glm::radians(gl_window_globals::yaw)) * cos(glm::radians(gl_window_globals::pitch))
        );
 
        model = glm::mat4(1.0f);

        projection = glm::perspective(glm::radians(gl_window_globals::fov), 
                                      (float)gl_window_globals::width / (float)gl_window_globals::height, 
                                      0.1f, gl_window_globals::far_plane_view_distance);  

        gl_window_globals::cameraFront = glm::normalize(direction);

        view = glm::lookAt(gl_window_globals::cameraPos, 
                           gl_window_globals::cameraPos + gl_window_globals::cameraFront, 
                           gl_window_globals::cameraUp);

        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));


        glClearColor(g::clear_color.x * g::clear_color.w, g::clear_color.y * g::clear_color.w, g::clear_color.z * g::clear_color.w, g::clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glUseProgram(shader_program);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBindVertexArray(VAO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, bodies.size() * sizeof(glm::vec3), float_body_positions.data());

        glPointSize(g::point_draw_size);

        glDrawArrays(GL_POINTS, 0, bodies.size()); // HERE 
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();

    return 0;
}

// MAIN
int main (int argc, char *argv[]) {
    // earth_moon_simulation(10000, 20, 365.25 * 20.0 * 24.0 * 3600.0);
    simulation_description three_body_example_simulation_description {
        generate_three_body_generator(12308045),
        0.0, 365.0 * 24.0 * 3600.0,
        0.5,
        600,
        timestep_RK4,
        calculate_gravitational_acceleration,
        brute_force_collision_resolution_velocity_change_calculation,
    };

    double earth_moon_num_seconds = 356.25 * 20.0 * 24.0 * 3600.0;
    double earth_moon_step_size = earth_moon_num_seconds / 100000.0;

    simulation_description earth_moon_simulation_description {
        earth_moon_bodies,
        0.0, earth_moon_num_seconds,
        earth_moon_step_size,
        earth_moon_step_size,
        timestep_RK4,
        calculate_gravitational_acceleration,
        collisions_dissabled
    };

    simulation_description thousand_bodies {
        generate_thousand_random_bodies,
        0.0, 365.25 * 24.0 * 3600.0,
        50,
        50,
        timestep_RK4,
        calculate_gravitational_acceleration,
        brute_force_collision_resolution_velocity_change_calculation
    };

    simulation_description two_body_collision {

    };

    // double days = 365.25 * 30.0;
    // three_body_simulation(0.0, days * 24.0 * 3600.0, 5.0*86400.0);
    //
    // three_body_simulation(three_body_example_simulation_description);
    // openglDisplay(three_body_example_simulation_description);
    openglDisplay(thousand_bodies);
    // earth_moon_simulation(earth_moon_simulation_description);
    return 0;
}


std::vector<simulationFrame> __run_nbody_simulation(
    size_t integration_steps, 
    size_t samples, 
    size_t length_simulation, 
    std::vector<gravitationalBody> bodies, 
    generic_integrator integration_method) 
{
    size_t sampling_divisor = integration_steps / samples;
    double a = 0, b = (length_simulation);
    double step_size = ((double)(b - a)) / ((double)integration_steps);

    integrator this_integrator(integration_method, calculate_gravitational_acceleration, collisions_dissabled, step_size);

    std::vector<simulationFrame> datalog;
    

    for (size_t s = 0; s < integration_steps; s++) {
        double target_time = a + step_size;
        a = this_integrator.integrate(bodies, a, target_time);

        if (s % sampling_divisor == 0) {
            double currentTime = a + s * step_size;
            datalog.push_back({bodies, currentTime});
        }
    }
    return datalog;
}

std::vector<simulationFrame> run_nbody_simulation(
    double sim_start,
    double sim_end,
    double step_size_hint,
    size_t samples,
    std::vector<gravitationalBody> bodies,
    generic_integrator integration_method)
{
    size_t num_integration_steps = (size_t)((sim_end - sim_start) / step_size_hint);
    integrator this_integrator = integrator(integration_method, calculate_gravitational_acceleration, collisions_dissabled, step_size_hint);

    return __run_nbody_simulation(num_integration_steps, samples, sim_end - sim_start, bodies, integration_method);
}


// std::vector<std::vector<simulationFrame>> three_body_simulation(simulation_description desc) {
//     const size_t num_bodies = 3;
//
//     std::vector<gravitationalBody> bodies = three_body_example_bodies();
//
//     std::vector<simulationFrame> rk2_datalog = run_nbody_simulation(desc.start, desc.end, desc.integrator_step_size_hint, 20, bodies, timestep_RK2);
//
//     std::vector<simulationFrame> rk4_datalog = run_nbody_simulation(desc.start, desc.end, desc.integrator_step_size_hint, 20, bodies, timestep_RK4);
//
//     std::println("3 body simulation");
//
//     std::println("{} Data Analysis", integrator::integrator_names[1]);
//     analyse_data_log(rk2_datalog);
//
//     std::println("{} Data Analysis", integrator::integrator_names[2]);
//     analyse_data_log(rk4_datalog);
//
//     return {rk2_datalog, rk4_datalog};
//
// }


std::vector<std::vector<simulationFrame>> earth_moon_simulation(size_t integration_steps, size_t samples, double num_seconds) {
    // const size_t num_bodies = 2;

    std::vector<gravitationalBody> bodies=earth_moon_bodies(); 

    double start = 0.0;
    double end = start + num_seconds;

    double step_size = (end - start) / ((double)(integration_steps));

    std::vector<simulationFrame> forward_euler_datalog = 
        run_nbody_simulation(start, end, step_size, samples, bodies, timestep_euler);

    std::vector<simulationFrame> rk2_datalog= 
        run_nbody_simulation(start, end, step_size, samples, bodies, timestep_RK2);
    
    std::vector<simulationFrame> rk4_datalog= 
        run_nbody_simulation(start, end, step_size, samples, bodies, timestep_RK4);
    
    std::println("2 body Earth moon simulation\n");
    std::println("{} Data Analysis", integrator::integrator_names[0]);
    analyse_data_log(forward_euler_datalog);

    std::println("{} Data Analysis", integrator::integrator_names[1]);
    analyse_data_log(rk2_datalog);

    std::println("{} Data Analysis", integrator::integrator_names[2]);
    analyse_data_log(rk4_datalog);

    return {forward_euler_datalog, rk2_datalog, rk4_datalog};
}


bool display_opengl_shader_compilation_error(unsigned int thisShader) {
    int sucess;
    char infoLog[512];
    glGetShaderiv(thisShader, GL_COMPILE_STATUS, &sucess);

    if (!sucess)
    {
        glGetShaderInfoLog(thisShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
        return true;
    }
    return false;
}

bool display_opengl_program_compilation_error(unsigned int program) {
    int sucess;
    char infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &sucess);
    if (!sucess) {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cout << "ERROR::PROGRAM::COMPILATION_FAILED\n" << infoLog << std::endl;
        return true;
    }
    return false;
};
