#include <algorithm>
#include <array>
#include <cmath>
#include <format>

#include <functional>
#include <glm/common.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/vec3.hpp>

#include <iterator>
#include <string_view>
#include <vector>
#include <iostream>

#include "analytics.h"


/* for n bodies in a system performs a pair wise computation to calculate the acceleration of each body 
 * mutates the second parameter the acceleration vec, note this will be resized to the size of the number
* of bodies in the system
 * */
void calculate_gravitational_acceleration(const std::vector<gravitationalBody> &bodies, std::vector<glm::dvec3>& acceleration) {
    acceleration.resize(bodies.size());
    std::fill(acceleration.begin(), acceleration.end(), glm::dvec3(0.0,0.0,0.0));

    double grav_const = GRAV_G;

    for (int i = 0; i < bodies.size()- 1; i++) {
        for (int j = i + 1; j < bodies.size(); j++) {
            glm::dvec3 pi = bodies[i].position, pj = bodies[j].position;
            double mi = bodies[i].mass, mj = bodies[j].mass;
            // vector pointing from j to i, the direction will the the same as the force i exertes on j
            glm::dvec3 dr = pi - pj;
            double dr_squared_magnitude = glm::dot(dr, dr);
            double inv_seperation_magnitude = 1.0 / std::sqrt(dr_squared_magnitude);
            glm::dvec3 force_on_j_direction = dr * inv_seperation_magnitude; // this is the normalised vector pointing from j to i
            glm::dvec3 force_on_i_direction= -force_on_j_direction;
            // acceleration of body 1 = G*M_2 / r^3
            acceleration[j] += (grav_const * mi / dr_squared_magnitude) * force_on_j_direction;
            acceleration[i] += (grav_const * mj / dr_squared_magnitude) * force_on_i_direction;
        }
    }
}
using accel_func = std::function<void 
    (const std::vector<gravitationalBody> &, 
     std::vector<glm::dvec3>&
)>;

/*
 * Forward euler n body integration function.
 * bodies : all gravitational bodies in the simulation
 * acceleration_junk_vec : this is essentially a vector containing the memory block used to store the acceleration 
 *   of each body once they have been computed
 * step_size : the delta t per step
 * acc_func : the function used to calculate the acceleration vectors
 * */

using forward_euler_function_signiture = std::function<void
    (std::vector<gravitationalBody>&,
     std::vector<glm::dvec3>&,
     double,
     accel_func
 )>;


/* Performs forward euler integration
 *
 * mutates 1st parameter
 * */
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

/*
 * Performs an runga kutta 2 timestep 
 * mutates 1st parameter
 * */
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

void _sum2_vec_dvec3(std::vector<glm::dvec3>& vec1, const std::vector<glm::dvec3>& vec2) {
    for (size_t i = 0; i < vec1.size(); i++) {
        vec1[i] += vec2[i]; 
    }
}

template<typename ...Vectors>
std::vector<glm::dvec3>& sum_vec_dvec(std::vector<glm::dvec3>& acc, Vectors const&... others) {
    (_sum2_vec_dvec3(acc, others), ...);
    return acc;
}

void timestep_RK4(
    std::vector<gravitationalBody>& bodies,
    std::array<std::vector<gravitationalBody>, 3>& body_copies,
    std::array<std::vector<glm::dvec3>, 4>& acceleration_junk,
    double step_size,
    accel_func acc_func,forward_euler_function_signiture forward_euler
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

    // acc_func computes acceleration from the state of the system
    // forward euler timesteps the system with the acceleration, by computing the acceleration of every body
    // it doesnt reset the passed acceleration itself, the passed acceleration function does this
    // therefore it is possible to break up the computation by computing acceleration outside of the forward euler method
    // acc_func(bodies, a1);

    forward_euler(x2, a1, step_size / 2.0, acc_func);
    
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


int main (int argc, char *argv[]) {
    size_t integration_steps = 200;
    size_t samples = 20;
    size_t sampling_divisor = integration_steps / samples;
    const size_t num_bodies = 2;
    double a = 0, b = (365.25f * 24.0f * 3600.0f);
    double step_size = ((double)(b - a)) / ((double)integration_steps);

    glm::dvec3 center_of_mass;
    std::cout << "step size : " << step_size << "\n";
    std::vector<glm::dvec3> acceleration;

    std::vector<gravitationalBody> bodies_original = {
        {MASS_EARTH, {0, 0,0},{0, 0,0}},
        {MASS_MOON, {0,DIST_EARTH_MOON,0},{MOON_EARTH_VELOCITY,0,0}},
    };

    center_of_mass = calculate_center_of_mass(bodies_original);
    acceleration.resize(bodies_original.size());
    std::fill(acceleration.begin(), acceleration.end(), glm::dvec3(0.0,0.0,0.0));

    std::cout << "Initial bodies: \n";
    for (size_t i = 0; i < bodies_original.size(); i++) {
        std::cout << std::format("{}, {}\n", get_str_glm_vec3(bodies_original[i].position, "position"), get_str_glm_vec3(bodies_original[i].velocity, "velocity"));
    }


    std::vector<simulationFrame<num_bodies>> dataLog_f_euler;
    std::vector<gravitationalBody> bodies_f_euler;
    std::copy(bodies_original.begin(), bodies_original.end(), std::back_inserter(bodies_f_euler));

    for (size_t s = 0; s < integration_steps; s++) {
        // calculate_gravitational_acceleration(bodies, acceleration);
        timestep_euler(bodies_f_euler, acceleration, step_size, calculate_gravitational_acceleration);

        if (s % sampling_divisor == 0) {
            int current_frame = s / sampling_divisor;
            double currentTime = a + s * step_size;

            dataLog_f_euler.push_back({});
            std::copy(bodies_f_euler.begin(), bodies_f_euler.end(), dataLog_f_euler[current_frame].bodies.begin());
            std::copy(acceleration.begin(), acceleration.end(), dataLog_f_euler[current_frame].acceleration.begin());
            dataLog_f_euler[current_frame].time = currentTime;
        }
    }
    std::cout << "\nForward Euler DATA ANALYSIS ===================\n";
    analyse_data_log(dataLog_f_euler);

    std::vector<simulationFrame<num_bodies>> datalog_RK2;
    std::vector<gravitationalBody> bodies_RK2;

    std::vector<gravitationalBody> bodies_RK2_copy;
    bodies_RK2_copy.resize(bodies_original.size());

    std::copy(bodies_original.begin(), bodies_original.end(), std::back_inserter(bodies_RK2));

    for (size_t s = 0; s < integration_steps; s++) {
        // calculate_gravitational_acceleration(bodies, acceleration);
        timestep_RK2(bodies_RK2, bodies_RK2_copy, acceleration, step_size, calculate_gravitational_acceleration, timestep_euler);

        if (s % sampling_divisor == 0) {
            int current_frame = s / sampling_divisor;
            double currentTime = a + s * step_size;

            datalog_RK2.push_back({});
            std::copy(bodies_RK2.begin(), bodies_RK2.end(), datalog_RK2[current_frame].bodies.begin());
            std::copy(acceleration.begin(), acceleration.end(), dataLog_f_euler[current_frame].acceleration.begin());
            datalog_RK2[current_frame].time = currentTime;
        }
    }
    std::cout << "\nRK2 DATA ANALYSIS ===================\n";
    analyse_data_log(datalog_RK2);

    std::vector<simulationFrame<num_bodies>> datalog_RK4;
    std::vector<gravitationalBody> bodies_RK4;

    std::array<std::vector<gravitationalBody>, 3> bodies_RK4_copy;
    for (auto& vec : bodies_RK4_copy) {
        vec.resize(bodies_original.size());
    }

    std::array<std::vector<glm::dvec3>, 4> acceleration_vecs_RK4;
    for (auto& acc: acceleration_vecs_RK4) {
        acc.resize(bodies_original.size());
    }

    std::copy(bodies_original.begin(), bodies_original.end(), std::back_inserter(bodies_RK4));

    for (size_t s = 0; s < integration_steps; s++) {
        // calculate_gravitational_acceleration(bodies, acceleration);
        timestep_RK4(bodies_RK4, bodies_RK4_copy, acceleration_vecs_RK4, step_size, calculate_gravitational_acceleration, timestep_euler);

        if (s % sampling_divisor == 0) {
            int current_frame = s / sampling_divisor;
            double currentTime = a + s * step_size;

            datalog_RK4.push_back({});
            std::copy(bodies_RK4.begin(), bodies_RK4.end(), datalog_RK4[current_frame].bodies.begin());
            std::copy(acceleration.begin(), acceleration.end(), dataLog_f_euler[current_frame].acceleration.begin());
            datalog_RK4[current_frame].time = currentTime;
        }
    }
    std::cout << "\nRK2 DATA ANALYSIS ===================\n";
    analyse_data_log(datalog_RK4);
    
    return 0;
}
