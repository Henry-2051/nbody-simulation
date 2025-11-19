#include "datatypes.h"
#include "integration_signitures.h"

#include<functional> 

#pragma once

using body_generator_function = std::function<std::vector<gravitationalBody>()>;

struct simulation_description {
    body_generator_function gen_bodies;
    double start, end;
    double step_size_hint;
    generic_integrator integrator;
    accel_func_signiture acceleration_function;
};

