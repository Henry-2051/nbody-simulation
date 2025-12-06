#include <glm/glm.hpp>
#include <string>
#include "integration_schemes.h"
#include <variant>


//////////////////////////////////////
/// Integrator interface signature ///

using generic_integrator = std::variant<forward_euler_function_signiture_interface, RK2_function_signiture, RK4_function_signiture>;
using generic_collision_res = std::variant<brute_force_col_res_func_sig, collisions_dissabled_func_sig>;


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
    const generic_integrator integration_method;
    const accel_func_signiture acceleration_function;
    const forward_euler_function_signiture_interface forward_euler;
    const generic_collision_res collision_resolution;

    std::vector<gravitationalBody> single_body_data;
    std::array<std::vector<gravitationalBody>, 3> triple_gravitational_body_data;

    std::vector<glm::dvec3> acceleration_junk_data;
    std::array<std::vector<glm::dvec3>, 4> acceleration_quadruple_junk_data;

    std::vector<glm::dvec3> velocity_change_junk;

    inline static std::vector<std::string> integrator_names {"Forward Euler", "Runge-Kutta 2nd Order", "Runge-Kutta 4th Order"};

    inline static std::vector<std::string> collision_res_names {"Brute force"};

    std::string integrator_name;

    double step_size;

    integrator(generic_integrator timestep_function, accel_func_signiture acc_func, generic_collision_res col_res, double step_size);
    integrator();

    void timestep_system(std::vector<gravitationalBody>& bodies, double step_size);
    double integrate(std::vector<gravitationalBody>& bodies, double current_time, double destination_time);
};
