#include "datatypes.h"
#include "integration_signitures.h"
#include "simulation_description.h"

#pragma once

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
