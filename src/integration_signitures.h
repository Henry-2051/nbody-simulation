#include "datatypes.h"
#include <variant>

#pragma once


#include <functional>
using accel_func_signiture = std::function<void 
    (const std::vector<gravitationalBody> &, 
     std::vector<glm::dvec3>&
)>;


using forward_euler_function_signiture_interface = std::function<void
    (std::vector<gravitationalBody>&,
     std::vector<glm::dvec3>&,
     double,
     accel_func_signiture
 )>;
using RK2_function_signiture = std::function<void(std::vector<gravitationalBody>&, std::vector<gravitationalBody>&, std::vector<glm::dvec3>&, double, accel_func_signiture, forward_euler_function_signiture_interface)>;

using RK4_function_signiture = std::function<void(std::vector<gravitationalBody>&, std::array<std::vector<gravitationalBody>, 3>&, std::array<std::vector<glm::dvec3>, 4>&, double, accel_func_signiture)>;

using generic_integrator = std::variant<forward_euler_function_signiture_interface, RK2_function_signiture, RK4_function_signiture>;

using brute_force_col_res_func_sig = std::function<void(const std::vector<gravitationalBody>&, std::vector<glm::dvec3>&, double)>;
using collisions_dissabled_func_sig = std::function<void()>;

using generic_collision_res = std::variant<brute_force_col_res_func_sig, collisions_dissabled_func_sig>;

/**
 * @brief Perform one explicit Euler integration step on a collection of bodies.
 *
 * This function advances the positions and velocities of a set of gravitational bodies
 * by a single time‐step Δt using the first‐order (explicit) Euler method:
 *   vᵢ ← vᵢ + aᵢ * Δt
 *   xᵢ ← xᵢ + vᵢ * Δt
 * where the accelerations {aᵢ} are supplied by the user’s acceleration function.
 *
 * @param bodies
 *   A vector of gravitationalBody instances representing the current state
 *   (mass, position, velocity) of each body.  On return, each body’s
 *   position and velocity have been updated in place.
 *
 * @param acceleration_junk
 *   A scratch buffer (of any size) that will be resized internally to match
 *   `bodies.size()`.  After calling the user’s acceleration function, it
 *   holds the acceleration vectors for each body:
 *     acceleration_junk[i] == acceleration of bodies[i]
 *
 * @param step_size
 *   The time increment Δt over which to step the simulation.  Units are
 *   arbitrary but must be consistent with those used by the acceleration function.
 *
 * @param acc_func
 *   A user‐provided function or callable object that computes accelerations for
 *   the current state of the system.
 *
 * @note
 *   - The explicit Euler scheme is only first‐order accurate and not symplectic,
 *     so it may exhibit energy drift over long simulations.
 *   - For better stability in gravitational N‐body problems, consider using
 *     a symplectic integrator (e.g., leap‐frog) or higher‐order Runge–Kutta.
 */
void timestep_euler(std::vector<gravitationalBody>& bodies, std::vector<glm::dvec3>& acceleration_junk, double step_size, accel_func_signiture acc_func);
 
/**
 * @brief Advance one time‐step using a 2nd‐order Runge–Kutta (RK2) integrator.
 *
 * Implements the explicit midpoint or Heun’s method:
 *   1. Compute k₁ =(tₙ,       yₙ)
 *   2. Estimate y* = yₙ + Δt·k₁
 *   3. Compute k₂ = f(tₙ + Δt,  y*)
 *   4. Update yₙ₊₁ = yₙ + (Δt/2)·(k₁ + k₂)
 *
 * Here y represents the full state (positions & velocities) of each body.
 *
 * @param bodies
 *   In/out vector of gravitationalBody.  On entry holds yₙ; on return holds yₙ₊₁.
 *
 * @param body_copy
 *   Scratch buffer (any size) for storing the intermediate state y*.  Will be
 *   resized internally to match `bodies.size()`.
 *
 * @param acceleration_junk
 *   Scratch buffer (any size) for storing acceleration vectors.  Will be resized
 *   internally to match `bodies.size()`.  Used to hold k₁ and k₂ accelerations.
 *
 * @param step_size
 *   Time‐step Δt for the integration.
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
 * The algorithm:
 *   k₁ = f(yₙ)
 *   k₂ = f(yₙ + Δt/2·k₁)
 *   k₃ = f(yₙ + Δt/2·k₂)
 *   k₄ = f(yₙ + Δt·k₃)
 *   yₙ₊₁ = yₙ + (Δt/6)(k₁ + 2k₂ + 2k₃ + k₄)
 *
 * Here y encapsulates all bodies’ positions and velocities.
 *
 * @param bodies
 *   In/out vector of gravitationalBody.  On entry holds yₙ; on return holds yₙ₊₁.
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
 * @param step_size
 *   Time‐step Δt for the integration.
 *
 * @param acc_func
 *   User‐supplied acceleration function (see `timestep_RK2` for signature).
 */
void timestep_RK4(
    std::vector<gravitationalBody>& bodies,
    std::array<std::vector<gravitationalBody>, 3>& body_copies,
    std::array<std::vector<glm::dvec3>, 4>& acceleration_junk,
    double step_size,
    accel_func_signiture acc_func
);

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
