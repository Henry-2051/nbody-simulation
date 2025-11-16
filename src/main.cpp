#include <algorithm>
#include <array>
#include <cmath>
#include <format>

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


#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "analytics.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

bool display_opengl_shader_compilation_error(unsigned int vertexShader); 
bool display_opengl_program_compilation_error(unsigned int program);

struct vertex_f {
    float x, y, z;
};

struct color_f {
    float r, g, b;
};

struct colored_vertex_f {
    vertex_f v;
    color_f  c;
};

unsigned int SCREEN_WIDTH = 800, SCREEN_HEIGHT = 600;

const char * vertex_shader_source = R"glsl(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aColor;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    out vec3 vertColor;

    void main()
    {
        gl_Position = projection * view * model * vec4(aPos, 1.0f);
        vertColor = aColor;
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


using accel_func = std::function<void 
    (const std::vector<gravitationalBody> &, 
     std::vector<glm::dvec3>&
)>;


using forward_euler_function_signiture = std::function<void
    (std::vector<gravitationalBody>&,
     std::vector<glm::dvec3>&,
     double,
     accel_func
 )>;
using RK2_function_signiture = std::function<void(std::vector<gravitationalBody>&, std::vector<gravitationalBody>&, std::vector<glm::dvec3>&, double, accel_func, forward_euler_function_signiture)>;

using RK4_function_signiture = std::function<void(std::vector<gravitationalBody>&, std::array<std::vector<gravitationalBody>, 3>&, std::array<std::vector<glm::dvec3>, 4>&, double, accel_func)>;

using generic_integrator = std::variant<forward_euler_function_signiture, RK2_function_signiture, RK4_function_signiture>;


/**
 * @brief Perform one explicit Euler integration step on a collection of bodies.
 *
 * This function advances the positions and velocities of a set of gravitational bodies
 * by a single time‐step Δt using the first‐order (explicit) Euler method:
 *   vᵢ ← vᵢ + aᵢ * Δt
 *   xᵢ ← xᵢ + vᵢ * Δt
 * where the accelerations {aᵢ} are supplied by the user’s acceleration function.
 *
 * @param bodies
 *   A vector of gravitationalBody instances representing the current state
 *   (mass, position, velocity) of each body.  On return, each body’s
 *   position and velocity have been updated in place.
 *
 * @param acceleration_junk
 *   A scratch buffer (of any size) that will be resized internally to match
 *   `bodies.size()`.  After calling the user’s acceleration function, it
 *   holds the acceleration vectors for each body:
 *     acceleration_junk[i] == acceleration of bodies[i]
 *
 * @param step_size
 *   The time increment Δt over which to step the simulation.  Units are
 *   arbitrary but must be consistent with those used by the acceleration function.
 *
 * @param acc_func
 *   A user‐provided function or callable object that computes accelerations for
 *   the current state of the system.
 *
 * @note
 *   - The explicit Euler scheme is only first‐order accurate and not symplectic,
 *     so it may exhibit energy drift over long simulations.
 *   - For better stability in gravitational N‐body problems, consider using
 *     a symplectic integrator (e.g., leap‐frog) or higher‐order Runge–Kutta.
 */
void timestep_euler(std::vector<gravitationalBody>& bodies, std::vector<glm::dvec3>& acceleration_junk, double step_size, accel_func acc_func);
 
/**
 * @brief Advance one time‐step using a 2nd‐order Runge–Kutta (RK2) integrator.
 *
 * Implements the explicit midpoint or Heun’s method:
 *   1. Compute k₁ =(tₙ,       yₙ)
 *   2. Estimate y* = yₙ + Δt·k₁
 *   3. Compute k₂ = f(tₙ + Δt,  y*)
 *   4. Update yₙ₊₁ = yₙ + (Δt/2)·(k₁ + k₂)
 *
 * Here y represents the full state (positions & velocities) of each body.
 *
 * @param bodies
 *   In/out vector of gravitationalBody.  On entry holds yₙ; on return holds yₙ₊₁.
 *
 * @param body_copy
 *   Scratch buffer (any size) for storing the intermediate state y*.  Will be
 *   resized internally to match `bodies.size()`.
 *
 * @param acceleration_junk
 *   Scratch buffer (any size) for storing acceleration vectors.  Will be resized
 *   internally to match `bodies.size()`.  Used to hold k₁ and k₂ accelerations.
 *
 * @param step_size
 *   Time‐step Δt for the integration.
 *
 * @param acc_func
 *   User‐supplied acceleration function with signature:
 *     void acc_func(
 *       const std::vector<gravitationalBody>& inBodies,
 *       std::vector<glm::dvec3>&             outAccelerations
 *     );
 *   Fills `outAccelerations[i] = aᵢ` for each body.
 *
 * @param forward_euler
 *   A callable matching the forward‐Euler step:
 *     void forward_euler(
 *       const std::vector<gravitationalBody>& in,
 *       std::vector<gravitationalBody>&       out,
 *       const std::vector<glm::dvec3>&        acc,
 *       double                                dt
 *     );
 *   Used internally to compute the intermediate state y*.
 */
void timestep_RK2(
    std::vector<gravitationalBody>& bodies, 
    std::vector<gravitationalBody>& body_copy, 
    std::vector<glm::dvec3>& acceleration_junk, 
    double step_size, 
    accel_func acc_func,
    forward_euler_function_signiture forward_euler);

void _sum2_vec_dvec3(std::vector<glm::dvec3>& vec1, const std::vector<glm::dvec3>& vec2);
void _scalarMul2_vec_dvec3(std::vector<glm::dvec3>& vec1, double scalar);
 
/**
 * @brief Advance one time‐step using the classical 4th‐order Runge–Kutta (RK4).
 *
 * The algorithm:
 *   k₁ = f(yₙ)
 *   k₂ = f(yₙ + Δt/2·k₁)
 *   k₃ = f(yₙ + Δt/2·k₂)
 *   k₄ = f(yₙ + Δt·k₃)
 *   yₙ₊₁ = yₙ + (Δt/6)(k₁ + 2k₂ + 2k₃ + k₄)
 *
 * Here y encapsulates all bodies’ positions and velocities.
 *
 * @param bodies
 *   In/out vector of gravitationalBody.  On entry holds yₙ; on return holds yₙ₊₁.
 *
 * @param body_copies
 *   Array of three scratch buffers for the intermediate states:
 *     body_copies[0] ← yₙ + (Δt/2)·k₁  
 *     body_copies[1] ← yₙ + (Δt/2)·k₂  
 *     body_copies[2] ← yₙ + Δt·k₃  
 *   Each vector will be resized internally to match `bodies.size()`.
 *
 * @param acceleration_junk
 *   Array of four scratch buffers for accelerations k₁…k₄.  Each vector will be
 *   resized internally to match `bodies.size()`.
 *
 * @param step_size
 *   Time‐step Δt for the integration.
 *
 * @param acc_func
 *   User‐supplied acceleration function (see `timestep_RK2` for signature).
 */
void timestep_RK4(
    std::vector<gravitationalBody>& bodies,
    std::array<std::vector<gravitationalBody>, 3>& body_copies,
    std::array<std::vector<glm::dvec3>, 4>& acceleration_junk,
    double step_size,
    accel_func acc_func
);

std::vector<glm::dvec3>& operator+=(std::vector<glm::dvec3>& vec1, const std::vector<glm::dvec3>& vec2);
std::vector<glm::dvec3>& operator*=(std::vector<glm::dvec3>& vec1, double scalar);

/**
 * @brief A polymorphic N-body integrator selecting among Euler, RK2, and RK4.
 *
 * This class wraps multiple time‐integration schemes for gravitationalBody systems.
 * The user provides:
 *   - a variant <generic_integrator> indicating which integrator to use,
 *   - an acceleration function <acceleration_function>,
 *   - a time‐step size <step_size>.
 *
 * Internally it allocates and reuses scratch buffers for intermediate states
 * and acceleration arrays appropriate to the chosen method.  Calling `integrate(bodies)`
 * dispatches to the selected scheme.
 */
struct integrator 
{
    const generic_integrator integration_method;
    const accel_func acceleration_function;
    const forward_euler_function_signiture forward_euler;

    std::vector<gravitationalBody> single_body_data;
    std::array<std::vector<gravitationalBody>, 3> triple_gravitational_body_data;

    std::vector<glm::dvec3> acceleration_junk_data;
    std::array<std::vector<glm::dvec3>, 4> acceleration_quadruple_junk_data;

    inline static std::vector<std::string> integrator_names {"Forward Euler", "Runge-Kutta 2nd Order", "Runge-Kutta 4th Order"};

    std::string integrator_name;

    double step_size;


    integrator(generic_integrator timestep_function, accel_func acc_func, double step_size): 
        integration_method(timestep_function), 
        acceleration_function(acc_func),
        forward_euler(timestep_euler),
        step_size(step_size)
    {
        if (std::holds_alternative<forward_euler_function_signiture>(integration_method)) 
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

    void integrate(std::vector<gravitationalBody>& bodies) {
        if (std::holds_alternative<forward_euler_function_signiture>(integration_method)) 
        {
            forward_euler_function_signiture euler_integrator = std::get<forward_euler_function_signiture>(integration_method);
            euler_integrator(bodies, acceleration_junk_data, step_size, acceleration_function);
        } 
        else if (std::holds_alternative<RK2_function_signiture>(integration_method)) 
        {
            RK2_function_signiture rk2_integrator = std::get<RK2_function_signiture>(integration_method);
            rk2_integrator(bodies, single_body_data, acceleration_junk_data, step_size, acceleration_function, forward_euler);
        } 
        else if (std::holds_alternative<RK4_function_signiture>(integration_method))
        {
            RK4_function_signiture rk4_integrator = std::get<RK4_function_signiture>(integration_method);
            rk4_integrator(bodies, triple_gravitational_body_data, acceleration_quadruple_junk_data, step_size, acceleration_function);
        }
    }
};

std::vector<simulationFrame> run_nbody_simulation(
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
        this_integrator.integrate(bodies);

        if (s % sampling_divisor == 0) {
            double currentTime = a + s * step_size;
            datalog.push_back({bodies, currentTime});
        }
    }
    return datalog;
}

template <size_t num_bodies>
std::vector<simulationFrame> run_nbody_simulation_prime(
    double sim_start,
    double sim_end,
    double step_size,
    size_t samples,
    std::vector<gravitationalBody> bodies,
    generic_integrator integration_method)
{
    size_t num_integration_steps = (size_t)((sim_end - sim_start) / step_size);
    integrator this_integrator = integrator(integration_method, calculate_gravitational_acceleration, step_size);

    return run_nbody_simulation(num_integration_steps, samples, sim_end - sim_start, bodies, integration_method);
}

std::vector<std::vector<simulationFrame>> three_body_simulation(double sim_start, double sim_end, double step_size) {
    const size_t num_bodies = 3;

    float angle1[2] = {30.0, 20.0 };
    float angle2[2] = {-20.0, 80.0};
    float angle3[2] = {60, 110};

    glm::vec3 x(1,0,0), y(0,1,0), z(0,0,1);

    // first chain of rotations
    glm::mat4 R1_a = glm::rotate(glm::mat4(1.0f),
                                 glm::radians(angle1[0]),
                                 x);
    glm::mat4 R1_b = glm::rotate(glm::mat4(1.0f),
                                 glm::radians(angle1[1]),
                                 y);
    // second chain
    glm::mat4 R2_a = glm::rotate(glm::mat4(1.0f),
                                 glm::radians(angle2[0]),
                                 x);
    glm::mat4 R2_b = glm::rotate(glm::mat4(1.0f),
                                 glm::radians(angle2[1]),
                                 y);
    // third chain
    glm::mat4 R3_a = glm::rotate(glm::mat4(1.0f),
                                 glm::radians(angle3[0]),
                                 x);
    glm::mat4 R3_b = glm::rotate(glm::mat4(1.0f),
                                 glm::radians(angle3[1]),
                                 y);

    glm::vec4 _r1 = R1_a * R1_b * glm::vec4(z, 0.0f);
    glm::dvec3 r1 = glm::dvec3(_r1);

    glm::vec3 _r2 = R2_b * R2_a * glm::vec4(z, 0.0);
    glm::dvec3 r2 = glm::dvec3(_r2);

    glm::vec3 _r3 = R3_b * R3_a * glm::vec4(z, 0.0);
    glm::dvec3 r3 = glm::dvec3(_r3);

    double large_dist = DIST_EARTH_MOON * 100.0f;
    double large_speed = MOON_EARTH_VELOCITY * 10.0;
    double large_mass = MASS_MARS;

    std::vector<gravitationalBody> bodies = {
        {large_mass * 1.32, large_dist * r1, large_speed * r2},
        {large_mass * 0.82, large_dist * r2, large_speed * r3},
        {large_mass * 1.58, large_dist * r3, large_speed * r1}
    };

    std::vector<simulationFrame> rk2_datalog = run_nbody_simulation_prime<num_bodies>(sim_start, sim_end, step_size, 20, bodies, timestep_RK2);
    
    std::vector<simulationFrame> rk4_datalog = run_nbody_simulation_prime<num_bodies>(sim_start, sim_end, step_size, 20, bodies, timestep_RK4);

    std::println("3 body simulation");

    std::println("{} Data Analysis", integrator::integrator_names[1]);
    analyse_data_log(rk2_datalog);

    std::println("{} Data Analysis", integrator::integrator_names[2]);
    analyse_data_log(rk4_datalog);

    return {rk2_datalog, rk4_datalog};
}

std::vector<std::vector<simulationFrame>> earth_moon_simulation(size_t integration_steps, size_t samples, double number_of_days) {
    const size_t num_bodies = 2;

    std::vector<gravitationalBody> bodies= {
        {MASS_EARTH, {0, 0,0},{0, 0,0}},
        {MASS_MOON, {0,DIST_EARTH_MOON,0},{MOON_EARTH_VELOCITY,0,0}},
    };


    std::vector<simulationFrame> forward_euler_datalog = 
        run_nbody_simulation(integration_steps, samples, number_of_days, bodies, timestep_euler);

    std::vector<simulationFrame> rk2_datalog= 
        run_nbody_simulation(integration_steps, samples, number_of_days, bodies, timestep_RK2);
    
    std::vector<simulationFrame> rk4_datalog= 
        run_nbody_simulation(integration_steps, samples, number_of_days, bodies, timestep_RK4);
    
    std::println("2 body Earth moon simulation\n");
    std::println("{} Data Analysis", integrator::integrator_names[0]);
    analyse_data_log(forward_euler_datalog);

    std::println("{} Data Analysis", integrator::integrator_names[1]);
    analyse_data_log(rk2_datalog);

    std::println("{} Data Analysis", integrator::integrator_names[2]);
    analyse_data_log(rk4_datalog);

    return {forward_euler_datalog, rk2_datalog, rk4_datalog};
}


int main (int argc, char *argv[]) {
    earth_moon_simulation(10000, 20, 365.25 * 20.0 * 24.0 * 3600.0);
    double days = 365.25 * 30.0;
    // three_body_simulation(0.0, days * 24.0 * 3600.0, 5.0*86400.0);
    return 0;
}

void timestep_euler(std::vector<gravitationalBody>& bodies, std::vector<glm::dvec3>& acceleration_junk, double step_size, accel_func acc_func) {
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
    accel_func acc_func,
    forward_euler_function_signiture forward_euler) 
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
    accel_func acc_func
)
{
    for (int i = 0; i < 4; i++) {
        acceleration_junk[i].resize(bodies.size());
        if (i != 3) {
            body_copies[i].resize(bodies.size());
            std::copy(bodies.begin(), bodies.end(), body_copies[i].begin());
        }
    }
    accel_func dummy_acceleration_function = [](const std::vector<gravitationalBody>& bodies, std::vector<glm::dvec3>& acceleration) {};

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
