#include "integrator.h"
#include <iostream>


integrator::integrator(generic_integrator timestep_function, accel_func_signiture acc_func, generic_collision_res col_res, double step_size): 
    integration_method(timestep_function), 
    acceleration_function(acc_func),
    forward_euler(timestep_euler),
    step_size(step_size),
    collision_resolution(col_res)
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

integrator::integrator() {}

void 
integrator::timestep_system(std::vector<gravitationalBody>& bodies, double deltaT) {
    // make sure the impulse vector for collisions is zero intialised 
    if (velocity_change_junk.size() != bodies.size()) {
        velocity_change_junk.resize(bodies.size());
        std::fill(velocity_change_junk.begin(), velocity_change_junk.end(), glm::dvec3(0.0,0.0,0.0));
    }

    if (std::holds_alternative<brute_force_col_res_func_sig>(collision_resolution)) {
        brute_force_col_res_func_sig col_res = std::get<brute_force_col_res_func_sig>(collision_resolution);
        col_res(bodies, velocity_change_junk, deltaT);
    } else if (std::holds_alternative<collisions_dissabled_func_sig>(collision_resolution)) {}

    for (int i = 0; i < bodies.size(); i++) {
        bodies[i].velocity += velocity_change_junk[i];
    }

    if (std::holds_alternative<forward_euler_function_signiture_interface>(integration_method)) 
    {
        forward_euler_function_signiture_interface euler_integrator = std::get<forward_euler_function_signiture_interface>(integration_method);
        euler_integrator(bodies, acceleration_junk_data, deltaT, acceleration_function);
    } 
    else if (std::holds_alternative<RK2_function_signiture>(integration_method)) 
    {
        RK2_function_signiture rk2_integrator = std::get<RK2_function_signiture>(integration_method);
        rk2_integrator(bodies, single_body_data, acceleration_junk_data, deltaT, acceleration_function, forward_euler);
    } 
    else if (std::holds_alternative<RK4_function_signiture>(integration_method))
    {
        RK4_function_signiture rk4_integrator = std::get<RK4_function_signiture>(integration_method);
        rk4_integrator(bodies, triple_gravitational_body_data, acceleration_quadruple_junk_data, deltaT, acceleration_function);
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
