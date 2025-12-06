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
    generic_integrator integrator;
    accel_func_signiture acceleration_function;
    generic_collision_res collision_function;
};

std::vector<simulationFrame> __run_nbody_simulation(
    size_t integration_steps, 
    size_t samples, 
    size_t length_simulation, 
    std::vector<gravitationalBody> bodies, 
    generic_integrator integration_method
); 

std::vector<simulationFrame> run_nbody_simulation(
    double sim_start,
    double sim_end,
    double step_size,
    size_t samples,
    std::vector<gravitationalBody> bodies,
    generic_integrator integration_method
);

std::vector<std::vector<simulationFrame>> three_body_simulation(simulation_description desc);

std::vector<std::vector<simulationFrame>> earth_moon_simulation(simulation_description desc);
