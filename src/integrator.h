#include "running_sim_signitures.h"
#include "datatypes.h"
#include <iostream>
#include <string>

#pragma once

struct integrator 
{
    const generic_integrator integration_method;
    const accel_func_signiture acceleration_function;
    const forward_euler_function_signiture_interface forward_euler;

    std::vector<gravitationalBody> single_body_data;
    std::array<std::vector<gravitationalBody>, 3> triple_gravitational_body_data;

    std::vector<glm::dvec3> acceleration_junk_data;
    std::array<std::vector<glm::dvec3>, 4> acceleration_quadruple_junk_data;

    inline static std::vector<std::string> integrator_names {"Forward Euler", "Runge-Kutta 2nd Order", "Runge-Kutta 4th Order"};

    std::string integrator_name;

    double step_size;

    integrator(generic_integrator timestep_function, accel_func_signiture acc_func, double step_size);

    double integrate(std::vector<gravitationalBody>& bodies, double current_time, double destination_time);
};
