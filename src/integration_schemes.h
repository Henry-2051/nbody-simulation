#include <glm/glm.hpp>
#include "primitiveDatatypes.h"

using accel_func_signiture = 
std::function<void 
    (const std::vector<gravitationalBody>&, std::vector<glm::dvec3>&)>;

using forward_euler_function_signiture_interface = std::function<void
    (std::vector<gravitationalBody>&,
     std::vector<glm::dvec3>&,
     double,
     accel_func_signiture
 )>;

using RK2_function_signiture = std::function<void(std::vector<gravitationalBody>&, std::vector<gravitationalBody>&, std::vector<glm::dvec3>&, double, accel_func_signiture, forward_euler_function_signiture_interface)>;

using RK4_function_signiture = std::function<void(std::vector<gravitationalBody>&, std::array<std::vector<gravitationalBody>, 3>&, std::array<std::vector<glm::dvec3>, 4>&, double, accel_func_signiture)>;

using brute_force_col_res_func_sig = std::function<void(const std::vector<gravitationalBody>&, std::vector<glm::dvec3>&, double)>;
using collisions_dissabled_func_sig = std::function<void()>;

void
inverse_cube_collision_approximation(
    const gravitationalBody& body1, const gravitationalBody& body2, 
    glm::dvec3& velocity_change1, glm::dvec3& velocity_change2,
    const glm::dvec3& norm_diff_r0, double diff_r0_magnitude, double step_size);


bool
quadratic_collision_detection_and_resolution(const gravitationalBody& body1, const gravitationalBody& body2,
                                             glm::dvec3& velocity_change1, glm::dvec3& velocity_change2,
                                             glm::dvec3& diff_r0, glm::dvec3& diff_v, double step_size);

void brute_force_collision_resolution_velocity_change_calculation(
    const std::vector<gravitationalBody>& bodies, std::vector<glm::dvec3>& velocity_change_junk, double step_size);

void calculate_gravitational_acceleration(const std::vector<gravitationalBody> &bodies, std::vector<glm::dvec3>& acceleration);

/**
 * @brief Perform one explicit Euler integration step on a collection of bodies.
 *
 * This function advances the positions and velocities of a set of gravitational bodies
 * by a single time‐step Δt using the first‐order (explicit) Euler method:
 *
 * @param acc_func
 *   User‐supplied acceleration function with signature:
 *     void acc_func(
 *       const std::vector<gravitationalBody>& inBodies,
 *       std::vector<glm::dvec3>&             outAccelerations
 *     );
 *   Fills `outAccelerations[i] = aᵢ` for each body.
 *
 * @param acceleration_junk
 *   A scratch buffer (of any size) that will be resized internally to match
 *   `bodies.size()`.  After calling the user’s acceleration function, it
 *   holds the acceleration vectors for each body:
 *     acceleration_junk[i] == acceleration of bodies[i]
 *
 *
 * @note
 *   - The explicit Euler scheme is only first‐order accurate and not symplectic,
 *     so it may exhibit energy drift over long simulations.
 */
void timestep_euler(std::vector<gravitationalBody>& bodies, std::vector<glm::dvec3>& acceleration_junk, double step_size, accel_func_signiture acc_func);
 
/**
 * @brief Advance one time‐step using a 2nd‐order Runge–Kutta (RK2) integrator.
 *
 * Implements the explicit midpoint or Heun’s method:
 *
 * @param body_copy
 *   Scratch buffer (any size) for storing the intermediate state y*.  Will be
 *   resized internally to match `bodies.size()`.
 *
 * @param acceleration_junk
 *   Scratch buffer (any size) for storing acceleration vectors.  Will be resized
 *   internally to match `bodies.size()`.  Used to hold k₁ and k₂ accelerations.
 *
 * @param acc_func
 *   User‐supplied acceleration function with signature:
 *     void acc_func(
 *       const std::vector<gravitationalBody>& inBodies,
 *       std::vector<glm::dvec3>&             outAccelerations
 *     );
 *   Fills `outAccelerations[i] = aᵢ` for each body.
 *
 * @param forward_euler
 *   A callable matching the forward‐Euler step:
 *     void forward_euler(
 *       const std::vector<gravitationalBody>& in,
 *       std::vector<gravitationalBody>&       out,
 *       const std::vector<glm::dvec3>&        acc,
 *       double                                dt
 *     );
 *   Used internally to compute the intermediate state y*.
 */
void timestep_RK2(
    std::vector<gravitationalBody>& bodies, 
    std::vector<gravitationalBody>& body_copy, 
    std::vector<glm::dvec3>& acceleration_junk, 
    double step_size, 
    accel_func_signiture acc_func,
    forward_euler_function_signiture_interface forward_euler);

void _sum2_vec_dvec3(std::vector<glm::dvec3>& vec1, const std::vector<glm::dvec3>& vec2);
void _scalarMul2_vec_dvec3(std::vector<glm::dvec3>& vec1, double scalar);
 
/**
 * @brief Advance one time‐step using the classical 4th‐order Runge–Kutta (RK4).
 *
 * @param body_copies
 *   Array of three scratch buffers for the intermediate states:
 *     body_copies[0] ← yₙ + (Δt/2)·k₁  
 *     body_copies[1] ← yₙ + (Δt/2)·k₂  
 *     body_copies[2] ← yₙ + Δt·k₃  
 *   Each vector will be resized internally to match `bodies.size()`.
 *
 * @param acceleration_junk
 *   Array of four scratch buffers for accelerations k₁…k₄.  Each vector will be
 *   resized internally to match `bodies.size()`.
 *
 * @param acc_func
 *   User‐supplied acceleration function with signature:
 *     void acc_func(
 *       const std::vector<gravitationalBody>& inBodies,
 *       std::vector<glm::dvec3>&             outAccelerations
 *     );
 *   Fills `outAccelerations[i] = aᵢ` for each body.
 */
void timestep_RK4(
    std::vector<gravitationalBody>& bodies,
    std::array<std::vector<gravitationalBody>, 3>& body_copies,
    std::array<std::vector<glm::dvec3>, 4>& acceleration_junk,
    double step_size,
    accel_func_signiture acc_func
);

// COLLISIONS

void calculate_spherical_radius(std::vector<gravitationalBody>& bodies, const std::vector<double>& densities);

void collisions_dissabled();

void
inverse_cube_collision_approximation(
    const gravitationalBody& body1, const gravitationalBody& body2, 
    glm::dvec3& velocity_change1, glm::dvec3& velocity_change2,
    const glm::dvec3& norm_diff_r0, double diff_r0_magnitude, double step_size);

bool
quadratic_collision_detection_and_resolution(const gravitationalBody& body1, const gravitationalBody& body2,
                                             glm::dvec3& velocity_change1, glm::dvec3& velocity_change2,
                                             glm::dvec3& diff_r0, glm::dvec3& diff_v, double step_size);

void brute_force_collision_resolution_velocity_change_calculation(
    const std::vector<gravitationalBody>& bodies, 
    std::vector<glm::dvec3>& velocity_change_junk, 
    double step_size);

