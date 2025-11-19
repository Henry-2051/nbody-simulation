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
#include <thread>
#include <vector>
#include <iostream>
#include <variant>
#include <chrono>
#include "integration_signitures.h"
#include "datatypes.h"

#include "integrator.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "analytics.h"
#include "running_sim_signitures.h"
#include "simulation_description.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

bool display_opengl_shader_compilation_error(unsigned int vertexShader); 
bool display_opengl_program_compilation_error(unsigned int program);

std::vector<gravitationalBody> generate_thousand_random_bodies() {
    BoundingBox pos_box {{-1000,-1000,-1000}, {1000,1000,1000}};
    pos_box.min *= 100.0;
    pos_box.max *= 100.0;

    BoundingBox vel_box {{-0,-0,-0}, {0,0,0}};

    const std::size_t num_points      = 1000;
    uint32_t    seed   = 12345;
    double      mMin   = 1e5;
    double      mMax   = 1e10;

    std::vector<gravitationalBody> bodies = generateRandomBodies(pos_box, vel_box, num_points, seed, mMin, mMax);

    return bodies;
}

std::vector<gravitationalBody> earth_moon_bodies() {
    return {
        {MASS_EARTH, {0, 0,0},{0, 0,0}},
        {MASS_MOON, {0,DIST_EARTH_MOON,0},{MOON_EARTH_VELOCITY,0,0}},
    };
}

std::vector<gravitationalBody> three_body_example_bodies() {
    float angle1[2] = {30.0, 20.0 };
    float angle2[2] = {-20.0, 80.0};
    float angle3[2] = {60, 110};

    glm::vec3 x(1,0,0), y(0,1,0), z(0,0,1);

    // first chain of rotations
    glm::mat4 R1_a = glm::rotate(glm::mat4(1.0f), glm::radians(angle1[0]), x);
    glm::mat4 R1_b = glm::rotate(glm::mat4(1.0f), glm::radians(angle1[1]), y);
    // second chain
    glm::mat4 R2_a = glm::rotate(glm::mat4(1.0f), glm::radians(angle2[0]), x);
    glm::mat4 R2_b = glm::rotate(glm::mat4(1.0f), glm::radians(angle2[1]), y);
    // third chain
    glm::mat4 R3_a = glm::rotate(glm::mat4(1.0f), glm::radians(angle3[0]), x);
    glm::mat4 R3_b = glm::rotate(glm::mat4(1.0f), glm::radians(angle3[1]), y);

    glm::vec4 _r1 = R1_a * R1_b * glm::vec4(z, 0.0f);
    glm::dvec3 r1 = glm::dvec3(_r1);

    glm::vec3 _r2 = R2_b * R2_a * glm::vec4(z, 0.0);
    glm::dvec3 r2 = glm::dvec3(_r2);

    glm::vec3 _r3 = R3_b * R3_a * glm::vec4(z, 0.0);
    glm::dvec3 r3 = glm::dvec3(_r3);

    double large_dist = DIST_EARTH_MOON * 1.0f;
    double large_speed = MOON_EARTH_VELOCITY*1.0;
    double large_mass = MASS_MARS* 100.0;


    std::vector<gravitationalBody> bodies = {
        {large_mass * 0.32, large_dist * r1, large_speed * r2},
        {large_mass * 0.22, large_dist * r2, large_speed * r3},
        {large_mass * 1.58, large_dist * r3, large_speed * r1}
    };

    return bodies;
}


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
}




integrator::integrator(generic_integrator timestep_function, accel_func_signiture acc_func, double step_size): 
    integration_method(timestep_function), 
    acceleration_function(acc_func),
    forward_euler(timestep_euler),
    step_size(step_size)
{
    if (std::holds_alternative<forward_euler_function_signiture_interface>(integration_method)) 
    {
        integrator_name = integrator_names[0];
    } 
    else if (std::holds_alternative<RK2_function_signiture>(integration_method)) 
    {
        integrator_name = integrator_names[1];
    } 
    else if (std::holds_alternative<RK4_function_signiture>(integration_method))
    {
        integrator_name = integrator_names[2];
    } else 
    {
        std::cerr << "Error integrator object isnt properly initialised";
    }
}


double
integrator::integrate(std::vector<gravitationalBody>& bodies, double currentTime, double destinationTime) {
    std::function<void(void)> integration_step;
    if (std::holds_alternative<forward_euler_function_signiture_interface>(integration_method)) 
    {
        forward_euler_function_signiture_interface euler_integrator = std::get<forward_euler_function_signiture_interface>(integration_method);
        integration_step = [&bodies, this, &euler_integrator](){
            euler_integrator(bodies, acceleration_junk_data, step_size, acceleration_function);
        };
    } 
    else if (std::holds_alternative<RK2_function_signiture>(integration_method)) 
    {
        RK2_function_signiture rk2_integrator = std::get<RK2_function_signiture>(integration_method);
        integration_step = [&bodies, this, &rk2_integrator]() {
            rk2_integrator(bodies, single_body_data, acceleration_junk_data, step_size, acceleration_function, forward_euler);
        };
    } 
    else if (std::holds_alternative<RK4_function_signiture>(integration_method))
    {
        RK4_function_signiture rk4_integrator = std::get<RK4_function_signiture>(integration_method);
        integration_step = [&bodies, this, &rk4_integrator]() {
            rk4_integrator(bodies, triple_gravitational_body_data, acceleration_quadruple_junk_data, step_size, acceleration_function);
        };
    }

    while (currentTime <= destinationTime) {
        currentTime += step_size;
        integration_step();
    }

    return currentTime;
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
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    gl_window_globals::width = width;
    gl_window_globals::height= height;
    glViewport(0, 0, width, height);
}


int openglDisplay(simulation_description sim_desc) {
    unsigned int VBO, VAO, EBO, vertex_shader, fragment_shader, shader_program;

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

    double target_step_size = 600;


    // HARDCODED EARTH MOON

    // float draw_scale_factor = 1.0;

    double max_dimension = get_max_dimension(bodies);
    float draw_scale_factor = 1000.0 / max_dimension;
    
    std::println("scale factor calculated as : {}", draw_scale_factor);

    auto scale_float_positions = [&float_body_positions, draw_scale_factor]() {
        for (auto& pos : float_body_positions) {
            pos *= draw_scale_factor;
        }
    };

    recalc_float_pos();

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

    integrator this_integrator = integrator(sim_desc.integrator, sim_desc.acceleration_function, sim_desc.step_size_hint);
    
    std::chrono::duration<double> one_sec(1.0);
    auto last_frame = std::chrono::high_resolution_clock::now();
    double current_simulation_time = sim_desc.start;
    double combined_energy_last = calculate_gpe(bodies) + calculate_kinetic_energy(bodies);

    auto last_second = std::chrono::high_resolution_clock::now();
    float time_accumulator = 0;
    size_t count = 0;
    while((!glfwWindowShouldClose(window))) {
        count ++;
        float currentFrame = glfwGetTime();
        gl_window_globals::deltaTime = currentFrame - gl_window_globals::lastFrame;
        gl_window_globals::lastFrame = currentFrame;
        time_accumulator += gl_window_globals::deltaTime;
        if (count % 60 == 0) {
            std::cout << "hihi " << time_accumulator << "\n";
            time_accumulator = 0;
            double combined_energy_current = calculate_kinetic_energy(bodies) + calculate_gpe(bodies);
            double perc_energy_divergence = 100.0* (combined_energy_current - combined_energy_last) / combined_energy_last;
            std::println("percentage energy divergence : {}", perc_energy_divergence);
            combined_energy_last = combined_energy_current;
        }


        double target_time = current_simulation_time + target_step_size;

        current_simulation_time = this_integrator.integrate(bodies, current_simulation_time, target_time);
        recalc_float_pos();
        scale_float_positions();

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

        glClearColor(0.1, 0.05, 0.11, 1.0); // this basically sets the clear color color
        // glClear(GL_COLOR_BUFFER_BIT); // this actually clears the color buffer


        glUseProgram(shader_program);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBindVertexArray(VAO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, bodies.size() * sizeof(glm::vec3), float_body_positions.data());

        // glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0); // positions
        glPointSize(5.0f);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawArrays(GL_POINTS, 0, bodies.size());
        // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        // glDrawElements(GL_TRIANGLES, 3*6, GL_UNSIGNED_INT, 0);
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
        three_body_example_bodies,
        0.0, 365.25 * 1.0 * 24.0 * 3600.0,
        150,
        timestep_RK4,
        calculate_gravitational_acceleration,
    };

    double earth_moon_num_seconds = 356.25 * 20.0 * 24.0 * 3600.0;
    double earth_moon_step_size = earth_moon_num_seconds / 100000.0;

    simulation_description earth_moon_simulation_description {
        earth_moon_bodies,
        0.0, earth_moon_num_seconds,
        earth_moon_step_size,
        timestep_RK4,
        calculate_gravitational_acceleration,
    };

    simulation_description thousand_bodies {
        generate_thousand_random_bodies,
        0.0, 365.25 * 24.0 * 3600.0,
        50,
        timestep_euler,
        calculate_gravitational_acceleration,
    };

    // double days = 365.25 * 30.0;
    // three_body_simulation(0.0, days * 24.0 * 3600.0, 5.0*86400.0);
    //
    // three_body_simulation(three_body_example_simulation_description);
    openglDisplay(thousand_bodies);
    // earth_moon_simulation(earth_moon_simulation_description);
    return 0;
}



void timestep_euler(std::vector<gravitationalBody>& bodies, std::vector<glm::dvec3>& acceleration_junk, double step_size, accel_func_signiture acc_func) {
    acceleration_junk.resize(bodies.size());

    acc_func(bodies, acceleration_junk);
    std::vector<glm::dvec3> acceleration = std::move(acceleration_junk);

    for (size_t i = 0; i < bodies.size(); i++) {
        bodies[i].position += bodies[i].velocity * step_size; 
        bodies[i].velocity += acceleration[i] * step_size;
    }

    acceleration_junk = std::move(acceleration);
}
void timestep_RK2(
    std::vector<gravitationalBody>& bodies, 
    std::vector<gravitationalBody>& body_copy, 
    std::vector<glm::dvec3>& acceleration_junk, 
    double step_size, 
    accel_func_signiture acc_func,
    forward_euler_function_signiture_interface forward_euler) 
{
    body_copy.resize(bodies.size());
    acceleration_junk.resize(bodies.size());

    std::copy(bodies.begin(), bodies.end(), body_copy.begin());
    forward_euler(body_copy, acceleration_junk, step_size / 2.0l, acc_func);

    acc_func(body_copy, acceleration_junk);
    std::vector<glm::dvec3> acceleration = std::move(acceleration_junk);

    for (size_t i = 0; i < bodies.size(); i++) {
        bodies[i].position += body_copy[i].velocity * step_size;
        bodies[i].velocity += acceleration[i] * step_size;
    }

    acceleration_junk = std::move(acceleration);
}

void timestep_RK4(
    std::vector<gravitationalBody>& bodies,
    std::array<std::vector<gravitationalBody>, 3>& body_copies,
    std::array<std::vector<glm::dvec3>, 4>& acceleration_junk,
    double step_size,
    accel_func_signiture acc_func
)
{
    for (int i = 0; i < 4; i++) {
        acceleration_junk[i].resize(bodies.size());
        if (i != 3) {
            body_copies[i].resize(bodies.size());
            std::copy(bodies.begin(), bodies.end(), body_copies[i].begin());
        }
    }
    accel_func_signiture dummy_acceleration_function = [](const std::vector<gravitationalBody>& bodies, std::vector<glm::dvec3>& acceleration) {};

    auto& x2 = body_copies[0];
    auto& x3 = body_copies[1];
    auto& x4 = body_copies[2];

    auto& a1 = acceleration_junk[0]; 
    auto& a2 = acceleration_junk[1];
    auto& a3 = acceleration_junk[2];
    auto& a4 = acceleration_junk[3];

    // were actually doing 2 seperate forward euler steps for each stage of the runge kutta

    acc_func(bodies, a1);
    for (size_t i = 0; i < bodies.size(); i ++) {
        x2[i].position += bodies[i].velocity * (step_size / 2.0);
        x2[i].velocity += a1[i] * (step_size / 2.0);
    }
    
    acc_func(x2, a2);
    for (size_t i = 0; i < bodies.size(); i ++) {
        x3[i].position += x2[i].velocity * (step_size / 2.0);
        x3[i].velocity += a2[i] * (step_size / 2.0);
    }

    acc_func(x3, a3);
    for (size_t i = 0; i < bodies.size(); i ++) {
        x4[i].position += x3[i].velocity * (step_size);
        x4[i].velocity += a3[i] * (step_size);
    }

    acc_func(x4, a4);
    for (size_t i = 0; i < bodies.size(); i++) {
        bodies[i].position += (step_size / 6.0) * (bodies[i].velocity + 2.0 * x2[i].velocity + 2.0 * x3[i].velocity + x4[i].velocity);
        bodies[i].velocity += (step_size / 6.0) * (a1[i] + 2.0 * a2[i] + 2.0 * a3[i] + a4[i]);
    }
}

void _sum2_vec_dvec3(std::vector<glm::dvec3>& vec1, const std::vector<glm::dvec3>& vec2) {
    for (size_t i = 0; i < vec1.size(); i++) {
        vec1[i] += vec2[i]; 
    }
}

std::vector<glm::dvec3>& operator+=(std::vector<glm::dvec3>& vec1, const std::vector<glm::dvec3>& vec2) {
    _sum2_vec_dvec3(vec1, vec2);
    return vec1;
}

void _scalarMul2_vec_dvec3(std::vector<glm::dvec3>& vec1, double scalar) {
    for (size_t i = 0; i < vec1.size(); i++) {
        vec1[i] *= scalar;
    }
}

std::vector<glm::dvec3>& operator*=(std::vector<glm::dvec3>& vec1, double scalar) {
    _scalarMul2_vec_dvec3(vec1, scalar);
    return vec1;
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

    integrator this_integrator(integration_method, calculate_gravitational_acceleration, step_size);

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
    integrator this_integrator = integrator(integration_method, calculate_gravitational_acceleration, step_size_hint);

    return __run_nbody_simulation(num_integration_steps, samples, sim_end - sim_start, bodies, integration_method);
}


std::vector<std::vector<simulationFrame>> three_body_simulation(simulation_description desc) {
    const size_t num_bodies = 3;

    std::vector<gravitationalBody> bodies = three_body_example_bodies();

    std::vector<simulationFrame> rk2_datalog = run_nbody_simulation(desc.start, desc.end, desc.step_size_hint, 20, bodies, timestep_RK2);
    
    std::vector<simulationFrame> rk4_datalog = run_nbody_simulation(desc.start, desc.end, desc.step_size_hint, 20, bodies, timestep_RK4);

    std::println("3 body simulation");

    std::println("{} Data Analysis", integrator::integrator_names[1]);
    analyse_data_log(rk2_datalog);

    std::println("{} Data Analysis", integrator::integrator_names[2]);
    analyse_data_log(rk4_datalog);

    return {rk2_datalog, rk4_datalog};

}


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
