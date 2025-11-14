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
#include <print>
#include <string_view>
#include <vector>
#include <iostream>
#include <variant>

#include "analytics.h"


/* for n bodies in a system performs a pair wise computation to calculate the acceleration of each body 
 * mutates the second parameter the acceleration vec, note this will be resized to the size of the number
* of bodies in the system
 * */
void calculate_gravitational_acceleration(const std::vector<gravitationalBody> &bodies, std::vector<glm::dvec3>& acceleration) {
    acceleration.resize(bodies.size());
    std::fill(acceleration.begin(), acceleration.end(), glm::dvec3(0.0,0.0,0.0));

    double grav_const = GRAV_G;

    for (size_t i = 0; i < bodies.size()- 1; i++) {
        for (size_t j = i + 1; j < bodies.size(); j++) {
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

using RK2_function_signiture = std::function<void(std::vector<gravitationalBody>&, std::vector<gravitationalBody>&, std::vector<glm::dvec3>&, double, accel_func, forward_euler_function_signiture)>;

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

// void _sum2_vec_dvec3(std::vector<glm::dvec3>& vec1, const std::vector<glm::dvec3>& vec2) {
//     for (size_t i = 0; i < vec1.size(); i++) {
//         vec1[i] += vec2[i]; 
//     }
// }
//
// template<typename ...Vectors>
// std::vector<glm::dvec3>& sum_vec_dvec(std::vector<glm::dvec3>& acc, Vectors const&... others) {
//     (_sum2_vec_dvec3(acc, others), ...);
//     return acc;
// }

using RK4_function_signiture = std::function<void(std::vector<gravitationalBody>&, std::array<std::vector<gravitationalBody>, 3>&, std::array<std::vector<glm::dvec3>, 4>&, double, accel_func)>;

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

using generic_integrator = std::variant<forward_euler_function_signiture, RK2_function_signiture, RK4_function_signiture>;
/* This basically 
 * */

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


    integrator(generic_integrator timestep_function, accel_func acc_func): 
        integration_method(timestep_function), 
        acceleration_function(acc_func),
        forward_euler(timestep_euler)
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

    void integrate(std::vector<gravitationalBody>& bodies, double timestep) {
        if (std::holds_alternative<forward_euler_function_signiture>(integration_method)) 
        {
            forward_euler_function_signiture euler_integrator = std::get<forward_euler_function_signiture>(integration_method);
            euler_integrator(bodies, acceleration_junk_data, timestep, acceleration_function);
        } 
        else if (std::holds_alternative<RK2_function_signiture>(integration_method)) 
        {
            RK2_function_signiture rk2_integrator = std::get<RK2_function_signiture>(integration_method);
            rk2_integrator(bodies, single_body_data, acceleration_junk_data, timestep, acceleration_function, forward_euler);
        } 
        else if (std::holds_alternative<RK4_function_signiture>(integration_method))
        {
            RK4_function_signiture rk4_integrator = std::get<RK4_function_signiture>(integration_method);
            rk4_integrator(bodies, triple_gravitational_body_data, acceleration_quadruple_junk_data, timestep, acceleration_function);
        }
    }
};

template <size_t num_bodies>
std::vector<simulationFrame<num_bodies>> run_nbody_simulation(
    size_t integration_steps, 
    size_t samples, 
    size_t number_of_days, 
    std::vector<gravitationalBody> bodies, 
    generic_integrator integration_method) 
{
    size_t sampling_divisor = integration_steps / samples;
    double a = 0, b = (number_of_days * 24.0f * 3600.0f);
    double step_size = ((double)(b - a)) / ((double)integration_steps);

    integrator this_integrator(integration_method, calculate_gravitational_acceleration);

    std::vector<simulationFrame<num_bodies>> datalog;

    for (size_t s = 0; s < integration_steps; s++) {
        this_integrator.integrate(bodies, step_size);

        if (s % sampling_divisor == 0) {
            int current_frame = s / sampling_divisor;
            double currentTime = a + s * step_size;

            datalog.push_back({});
            std::copy(bodies.begin(), bodies.end(), datalog[current_frame].bodies.begin());
            datalog[current_frame].time = currentTime;
        }
    }
    return datalog;
}

std::vector<std::vector<simulationFrame<2>>> earth_moon_simulation(size_t integration_steps, size_t samples, double number_of_days) {
    const size_t num_bodies = 2;

    std::vector<gravitationalBody> bodies= {
        {MASS_EARTH, {0, 0,0},{0, 0,0}},
        {MASS_MOON, {0,DIST_EARTH_MOON,0},{MOON_EARTH_VELOCITY,0,0}},
    };

    std::vector<simulationFrame<num_bodies>> forward_euler_datalog = 
        run_nbody_simulation<num_bodies>(integration_steps, samples, number_of_days, bodies, timestep_euler);

    std::vector<simulationFrame<num_bodies>> rk2_datalog= 
        run_nbody_simulation<num_bodies>(integration_steps, samples, number_of_days, bodies, timestep_RK2);
    
    std::vector<simulationFrame<num_bodies>> rk4_datalog= 
        run_nbody_simulation<num_bodies>(integration_steps, samples, number_of_days, bodies, timestep_RK4);
    
    std::println("{} Data Analysis", integrator::integrator_names[0]);
    analyse_data_log(forward_euler_datalog);

    std::println("{} Data Analysis", integrator::integrator_names[1]);
    analyse_data_log(rk2_datalog);

    std::println("{} Data Analysis", integrator::integrator_names[2]);
    analyse_data_log(rk4_datalog);

    return {forward_euler_datalog, rk2_datalog, rk4_datalog};
}


int main (int argc, char *argv[]) {
    earth_moon_simulation(10000, 20, 365.25*20.0);
    return 0;
}
