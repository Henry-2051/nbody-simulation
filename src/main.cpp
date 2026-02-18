#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <memory>
#include <ostream>
#include <ratio>
#include <sstream>

#include <functional>
#include <glm/common.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/quaternion_transform.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/fwd.hpp>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <print>
#include <vector>
#include <iostream>
#include <variant>
#include <chrono>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "datatypes.h"
#include "analytics.h"

#include "main_opengl_render_structs.hpp"
#include "simulation_generator_functions.hpp"

#include "main_opengl_render.hpp"




// MAIN
int main (int argc, char *argv[]) {
    // earth_moon_simulation(10000, 20, 365.25 * 20.0 * 24.0 * 3600.0);
    simulation_description three_body_example_simulation_description {
        generate_three_body_generator(12308045),
        0.0, 365.0 * 24.0 * 3600.0,
        0.5,
        600,
        Integrator_Type::RungeKutta4,
        calculate_gravitational_acceleration,
        Col_Resolution_Type::BruteForce,
    };

    double earth_moon_num_seconds = 356.25 * 20.0 * 24.0 * 3600.0;
    double earth_moon_step_size = earth_moon_num_seconds / 100000.0;

    simulation_description earth_moon_simulation_description {
        earth_moon_bodies,
        0.0, earth_moon_num_seconds,
        earth_moon_step_size,
        earth_moon_step_size,
        Integrator_Type::RungeKutta4,
        calculate_gravitational_acceleration,
        Col_Resolution_Type::Dissabled
    };

    simulation_description thousand_bodies {
        generate_thousand_random_bodies,
        0.0, 365.25 * 24.0 * 3600.0,
        50,
        50,
        Integrator_Type::RungeKutta4,
        calculate_gravitational_acceleration,
       Col_Resolution_Type::BruteForce 
    };

    simulation_description fivehundread_uniform_random_bodies {
        generate_fivehundread_uniform_random_bodies,
        0.0, 365.25 * 24.0 * 3600.0,
        50,
        50,
        Integrator_Type::RungeKutta4,
        calculate_gravitational_acceleration,
        Col_Resolution_Type::BruteForce 

    };

    // double days = 365.25 * 30.0;
    // three_body_simulation(0.0, days * 24.0 * 3600.0, 5.0*86400.0);
    //
    // three_body_simulation(three_body_example_simulation_description);
    // openglDisplay(three_body_example_simulation_description);
    GLWindowGlobals win_globs {};
    SimulationGLobals sim_globs {};
    openglDisplay(fivehundread_uniform_random_bodies, std::move(win_globs), sim_globs);
    // openglDisplay(thousand_bodies);
    // earth_moon_simulation(earth_moon_simulation_description);
    return 0;
}

std::vector<simulationFrame> run_nbody_simulation(
    double sim_start,
    double sim_end,
    double step_size_hint,
    size_t samples,
    std::vector<gravitationalBody> bodies,
    Integrator_Type integrator_type,
    Col_Resolution_Type collision_resolution_type
) {
    size_t num_integration_steps = (size_t)((sim_end - sim_start) / step_size_hint);
    integrator this_integrator(integrator_type, collision_resolution_type, calculate_gravitational_acceleration, step_size_hint);

    size_t sampling_divisor = num_integration_steps/ samples;
    double step_size = (sim_end - sim_start) / ((double)num_integration_steps);

    std::vector<simulationFrame> datalog;

    double current_time = sim_start;

    for (size_t s = 0; s < num_integration_steps; s++) {
        double target_time = sim_start + step_size;
        current_time = this_integrator.integrate(bodies, current_time, target_time);

        if (s % sampling_divisor == 0) {
            datalog.push_back({bodies, current_time});
        }
    }

    return datalog;
}


// std::vector<std::vector<simulationFrame>> three_body_simulation(simulation_description desc) {
//     const size_t num_bodies = 3;
//
//     std::vector<gravitationalBody> bodies = three_body_example_bodies();
//
//     std::vector<simulationFrame> rk2_datalog = run_nbody_simulation(desc.start, desc.end, desc.integrator_step_size_hint, 20, bodies, timestep_RK2);
//
//     std::vector<simulationFrame> rk4_datalog = run_nbody_simulation(desc.start, desc.end, desc.integrator_step_size_hint, 20, bodies, timestep_RK4);
//
//     std::println("3 body simulation");
//
//     std::println("{} Data Analysis", integrator::integrator_names[1]);
//     analyse_data_log(rk2_datalog);
//
//     std::println("{} Data Analysis", integrator::integrator_names[2]);
//     analyse_data_log(rk4_datalog);
//
//     return {rk2_datalog, rk4_datalog};
//
// }


std::vector<std::vector<simulationFrame>> earth_moon_simulation(size_t integration_steps, size_t samples, double num_seconds) {
    // const size_t num_bodies = 2;

    std::vector<gravitationalBody> bodies=earth_moon_bodies(); 

    double start = 0.0;
    double end = start + num_seconds;

    double step_size = (end - start) / ((double)(integration_steps));

    std::vector<simulationFrame> forward_euler_datalog = 
        run_nbody_simulation(start, end, step_size, samples, bodies, Integrator_Type::ForwardEuler, Col_Resolution_Type::Dissabled);

    std::vector<simulationFrame> rk2_datalog= 
        run_nbody_simulation(start, end, step_size, samples, bodies, Integrator_Type::RungeKutta2, Col_Resolution_Type::Dissabled);
    
    std::vector<simulationFrame> rk4_datalog= 
        run_nbody_simulation(start, end, step_size, samples, bodies, Integrator_Type::RungeKutta4, Col_Resolution_Type::Dissabled);
    
    std::println("2 body Earth moon simulation\n");
    std::println("{} Data Analysis", integrator::integrator_names[0]);
    analyse_data_log(forward_euler_datalog);

    std::println("{} Data Analysis", integrator::integrator_names[1]);
    analyse_data_log(rk2_datalog);

    std::println("{} Data Analysis", integrator::integrator_names[2]);
    analyse_data_log(rk4_datalog);

    return {forward_euler_datalog, rk2_datalog, rk4_datalog};
}
