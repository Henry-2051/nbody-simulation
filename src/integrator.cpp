#include "integrator.h"
#include "datatypes.h"
#include <format>
#include <iostream>
#include <stdexcept>


integrator::integrator(Integrator_Type itt, Col_Resolution_Type col_res, accel_func_signiture acc_func, double step_size): 
    integration_type(itt),
    collision_resolution_type(col_res),
    acceleration_function(acc_func),
    step_size(step_size)
{
}

void 
integrator::timestep_system(std::vector<gravitationalBody>& bodies, double deltaT) {
    // make sure the impulse vector for collisions is zero intialised 
    if (velocity_change_junk.size() != bodies.size()) {
        velocity_change_junk.resize(bodies.size());
        std::fill(velocity_change_junk.begin(), velocity_change_junk.end(), glm::dvec3(0.0,0.0,0.0));
    }

    switch (collision_resolution_type) {
    case Col_Resolution_Type::BruteForce:
        brute_force_collision_resolution_velocity_change_calculation(bodies, velocity_change_junk, deltaT);
        break;
    case Col_Resolution_Type::Dissabled:
        break;
    }

    for (int i = 0; i < bodies.size(); i++) {
        bodies[i].velocity += velocity_change_junk[i];
    }

    switch (integration_type) {
    case Integrator_Type::ForwardEuler:
        timestep_euler(bodies, acceleration_junk_data, deltaT, acceleration_function);
        break;
    case Integrator_Type::SymplecticEuler:
        timestep_symplectic_euler(bodies, acceleration_junk_data, deltaT, acceleration_function);
        break;
    case Integrator_Type::RungeKutta2:
        timestep_RK2(bodies, single_body_data, acceleration_junk_data, deltaT, acceleration_function);
        break;
    case Integrator_Type::ImplicitMidpoint:
        timestep_implicit_midpoint(bodies, single_body_data, acceleration_junk_data, deltaT, acceleration_function);
        break;
    case Integrator_Type::RungeKutta4:
        timestep_RK4(bodies, triple_gravitational_body_data, acceleration_quadruple_junk_data, deltaT, acceleration_function);
        break;
    default:
        throw std::runtime_error(std::format("Invalid state, Integrator_Type has unhandled value (integrator::timestep_system in integrator.cpp) of {}", to_string(integration_type)));
    }
}

double
integrator::integrate(std::vector<gravitationalBody>& bodies, double start_time, double end_time) {
    double integration_period = end_time - start_time;
    size_t amount_steps = std::max(size_t(1), (size_t)(integration_period / step_size)) ;
    for (int i = 0; i < amount_steps; i++) {
        start_time += step_size;
        timestep_system(bodies, step_size);
    }
    return start_time;
}
