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

// describes a possible nbody simulation, 
// body_generator_function 
// start and end time in seconds
// integrator step size hint, this basically tells the integrator how much it should integrate at once, or rather the granularity of the numerical simulation
// simulation step size, when rendered this corresponds to how much time should pass per frame, there can be multiple integration steps per frame
// integrator_type, an enum that tells us which integration scheme to use, cant use direct dependency injection because for example a second order method would have
// more parameters than a first order method
// acceleration_function, the acceleration function (dependency injection)
// collision_resolution_type, same situation as the integrator_type
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
