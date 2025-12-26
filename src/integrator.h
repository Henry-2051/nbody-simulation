#pragma once  
#include <glm/glm.hpp>
#include <string>
#include "integration_schemes.h"
#include <variant>

//////////////////////////////////
///integrator enum

//////////////////////////////////////
/// Integrator interface signature ///


std::vector<glm::dvec3>& operator+=(std::vector<glm::dvec3>& vec1, const std::vector<glm::dvec3>& vec2);
std::vector<glm::dvec3>& operator*=(std::vector<glm::dvec3>& vec1, double scalar);


/**
 * @brief A polymorphic N-body integrator selecting among Euler, RK2, and RK4.
 *
 * This class wraps multiple time‐integration schemes for gravitationalBody systems.
 * The user provides:
 *   - a variant <generic_integrator> indicating which integrator to use,
 *   - an acceleration function <acceleration_function>,
 *   - a time‐step size <step_size>.
 *
 * Internally it allocates and reuses scratch buffers for intermediate states
 * and acceleration arrays appropriate to the chosen method.  Calling `integrate(bodies)`
 * dispatches to the selected scheme.
 */


struct integrator 
{
    const Integrator_Type integration_type;
    const Col_Resolution_Type collision_resolution_type;
    const accel_func_signiture acceleration_function;

    std::vector<gravitationalBody> single_body_data;
    std::array<std::vector<gravitationalBody>, 3> triple_gravitational_body_data;

    std::vector<glm::dvec3> acceleration_junk_data;
    std::array<std::vector<glm::dvec3>, 4> acceleration_quadruple_junk_data;

    std::vector<glm::dvec3> velocity_change_junk;

    inline static std::vector<std::string> integrator_names {"Forward Euler", "Runge-Kutta 2nd Order", "Runge-Kutta 4th Order", 
        "midpoint (symplectic) 2nd order", "symplectic euler"};

    inline static std::vector<std::string> collision_res_names {"Brute force"};

    std::string integrator_name;

    double step_size;

    integrator(Integrator_Type iit, Col_Resolution_Type c_res, accel_func_signiture acc_func, double step_size);
    integrator();

    void timestep_system(std::vector<gravitationalBody>& bodies, double step_size);
    double integrate(std::vector<gravitationalBody>& bodies, double current_time, double destination_time);
};
