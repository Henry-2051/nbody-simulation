#pragma once
#include <glm/glm.hpp>
#include <functional>
#include "integrator.h"
#include <variant>

struct simulationFrame {
    std::vector<gravitationalBody> bodies;
    double time;
    // by default we are going to assume that every element of the array 
    // contains a valid gravitational body datapoint, if the situation arrises where we want to test
    // something weird like add another gravitational body mid simulation we will have to take care 
    // of this.
};

using body_generator_function = std::function<std::vector<gravitationalBody>()>;

struct simulation_description {
    body_generator_function gen_bodies;
    double start, end;
    double integrator_step_size_hint;
    double simulation_step_size;
    Integrator_Type integrator_type;
    accel_func_signiture acceleration_function;
    Col_Resolution_Type collision_resolution_type;
};

std::vector<std::vector<simulationFrame>> three_body_simulation(simulation_description desc);

std::vector<std::vector<simulationFrame>> earth_moon_simulation(simulation_description desc);
