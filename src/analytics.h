#include <array>
#include <cmath>
#include <format>
#include <random>

#include <glm/common.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/vec3.hpp>

#include <print>
#include <string_view>
#include <vector>
#include <iostream>
// #include <Eigen/Eigen>
#include "datatypes.h"

#pragma once

#define GRAV_G 6.67408e-11
#define MASS_EARTH 5.9722e24
#define MASS_MOON 7.348e22
#define DIST_EARTH_MOON 3.84e8
#define MOON_ANG_FREQ 2.66e-6
#define MOON_EARTH_VELOCITY (MOON_ANG_FREQ * DIST_EARTH_MOON)
#define EARTH_SUN_VELOCITY 2.9886e5
#define MASS_MARS 6.417e23

void calculate_spherical_radius(std::vector<gravitationalBody>& bodies, const std::vector<double>& densities) {
    for (size_t i = 0; i < bodies.size(); i++) {
        bodies[i].radius = std::pow((3/(4.0*M_PI)) * (bodies[i].mass / densities[i]), 1.0/3.0);
    }
}

void collisions_dissabled() {}

void
inverse_cube_collision_approximation(
    const gravitationalBody& body1, const gravitationalBody& body2, 
    glm::dvec3& velocity_change1, glm::dvec3& velocity_change2,
    const glm::dvec3& norm_diff_r0, double diff_r0_magnitude, double step_size) 
{
    double beta_constant = - GRAV_G * body1.mass * body2.mass * (body1.radius + body2.radius);

    double force = beta_constant * (1 / std::pow(diff_r0_magnitude, 3) );
    velocity_change1 -= norm_diff_r0 * step_size * (force / body1.mass);
    velocity_change2 += norm_diff_r0 * step_size * (force / body2.mass);
}

bool
quadratic_collision_detection_and_resolution(const gravitationalBody& body1, const gravitationalBody& body2,
                                             glm::dvec3& velocity_change1, glm::dvec3& velocity_change2,
                                             glm::dvec3& diff_r0, glm::dvec3& diff_v, double step_size) 
{
    double sphere_rad_sum = body1.radius + body2.radius;

    // quadratic coefficients 
    double c = glm::dot(diff_r0, diff_r0) - sphere_rad_sum * sphere_rad_sum;
    double b = 2.0 * glm::dot(diff_r0, diff_v);
    double a = glm::dot(diff_v, diff_v);

    double determinant = b*b - 4.0*a*c;

    if (determinant <= 0)
        return false;

    double t1 = (-b + sqrt(determinant)) / (2.0 * a);
    double t2 = (-b - sqrt(determinant)) / (2.0 * a);

    auto is_time_in_window = [&step_size](double t) {
        return (t > 0.0 && t <= step_size);
    };

    double t;
    if (!is_time_in_window(t1) && !is_time_in_window(t2)) {
        return false;
    }

    if (!is_time_in_window(t1)) {
        t = t2;
    } else if (!is_time_in_window(t2)) {
        t = t1;
    } else {
        t = std::min(t1, t2);
    }

    // std::cout << std::format("collision {} \n", bodies[j].position.x);

    // outer_collision_res(t, diff_v, diff_r0, i, j);

    // now we have the time of collision we calculate the difference
    // this vector pointers from r2 to r1 at the time of collision
    // the direction point 1 will be going

    glm::dvec3 norm_diff_rt = glm::normalize(diff_r0 + t * diff_v);
    double rel_normal_velocity_change = -(1.0 + (body1.restitution * body2.restitution)) * glm::dot(diff_v, norm_diff_rt);

    double m1 = body1.mass;
    double m2 = body2.mass;

    glm::dvec3 delta_v1 = norm_diff_rt * ((m2*rel_normal_velocity_change) / (m1+m2));
    glm::dvec3 delta_v2 = norm_diff_rt * (-(m1*rel_normal_velocity_change) / (m1+m2));

    velocity_change1 += delta_v1;
    velocity_change2 += delta_v2;
    return true;
}

void brute_force_collision_resolution_velocity_change_calculation(
    const std::vector<gravitationalBody>& bodies, 
    std::vector<glm::dvec3>& velocity_change_junk, 
    double step_size) 
{
    velocity_change_junk.resize(bodies.size());
    std::fill(velocity_change_junk.begin(), velocity_change_junk.end(), glm::dvec3(0.0,0.0,0.0));

    for (size_t i = 0; i < (bodies.size() - 1); i++) {
        for (size_t j = i + 1; j < bodies.size(); j++) {
            double sum_mass = bodies[i].mass + bodies[j].mass;
            glm::dvec3 sum_momentum = bodies[i].velocity * bodies[i].mass + bodies[j].velocity * bodies[j].mass;
            glm::dvec3 vboost = sum_momentum / (sum_mass);
            glm::dvec3 m_bar = (bodies[i].position * bodies[i].mass + bodies[j].position * bodies[j].mass) / sum_mass;

            // we transition into the lorentz frame of the collision
            glm::dvec3 v1 = bodies[i].velocity - vboost;
            glm::dvec3 v2 = bodies[j].velocity - vboost;

            glm::dvec3 r10 = bodies[i].position - m_bar;
            glm::dvec3 r20 = bodies[j].position - m_bar;

            glm::dvec3 diff_r0 = r10 - r20;
            glm::dvec3 diff_v = v1 - v2;

            glm::dvec3 norm_diff_r0 = glm::normalize(diff_r0);
            double diff_r0_magnitude = std::sqrt(glm::dot(diff_r0, diff_r0));

            double overlap = bodies[i].radius + bodies[j].radius - diff_r0_magnitude;

            inverse_cube_collision_approximation(bodies[i], bodies[j], 
                         velocity_change_junk[i], velocity_change_junk[j], 
                         norm_diff_r0, diff_r0_magnitude, step_size);
            
            // inverse r^3 repulsion law, irrelevant over large distances but prevents bodies from overlapping at smaller distances
            // the force is calculated such that it is equal anad opposite to the gravitational attraction when the bodies are just barely
            // touching 
            // this also makes our quadratic collisions more accurate becasue when 2 bodies get close now the acceleration approaches zero 
            
            // we shouldnt execute the collision code when the bodies are overlapping because 
            if (overlap > 0.0) {
                std::cout << std::format("inside!!!{} \n", bodies[j].position.x);
                continue;
            }

            // continue of the bodies aren't moving towards each other
            if (glm::dot(diff_r0, diff_v) >= 0.0) {
                continue;
            }

            quadratic_collision_detection_and_resolution(bodies[i], bodies[j],
                             velocity_change_junk[i], velocity_change_junk[j], 
                             diff_r0, diff_v, step_size);
        }
    }
}

// A gravitational body.
// double mass  8 bytes
// glm::dvec3 position 24 bytes
// glm::dvec3 velocity 24 bytes

inline void calculate_gravitational_acceleration(const std::vector<gravitationalBody> &bodies, std::vector<glm::dvec3>& acceleration) {
    acceleration.resize(bodies.size());
    std::fill(acceleration.begin(), acceleration.end(), glm::dvec3(0.0,0.0,0.0));

    double grav_const = GRAV_G;

    for (size_t i = 0; i < bodies.size()- 1; i++) {
        for (size_t j = i + 1; j < bodies.size(); j++) {
            glm::dvec3 pi = bodies[i].position, pj = bodies[j].position;
            double mi = bodies[i].mass, mj = bodies[j].mass;
            // vector pointing from j to i, the direction will the the same as the force i exertes on j
            glm::dvec3 dr = pi - pj;
            double dr_squared_magnitude = glm::dot(dr, dr);
            double inv_seperation_magnitude = 1.0 / std::sqrt(dr_squared_magnitude);
            glm::dvec3 force_on_j_direction = dr * inv_seperation_magnitude; // this is the normalised vector pointing from j to i
            glm::dvec3 force_on_i_direction= -force_on_j_direction;
            // acceleration of body 1 = G*M_2 / r^3
            acceleration[j] += (grav_const * mi / dr_squared_magnitude) * force_on_j_direction;
            acceleration[i] += (grav_const * mj / dr_squared_magnitude) * force_on_i_direction;
        }
    }
}
inline std::ostream& operator<<(std::ostream& os, glm::dvec3 const& v)
{
    return os << '(' << v.x << ", " << v.y << ", " << v.z << ')';
}

// using collision_candidate_function_sig = std::function<void(size_t, const std::vector<gravitationalBody>&, std::vector<std::vector<size_t>>&)>;



/// generate N bodies with
///  - position uniformly in [box.min, box.max]
///  - velocity    uniformly in [box.min, box.max]
///  - mass        uniformly in [massMin, massMax]
inline std::vector<gravitationalBody>
generateRandomBodies(const BoundingBox &pos_box,
                     const BoundingBox &vel_box,
                     std::size_t         N,
                     uint32_t            seed,
                     double              massMin   = 1.0,
                     double              massMax   = 10.0)
{
    // rng
    std::mt19937_64 rng(seed);

    // position distributions
    std::uniform_real_distribution<double> distX(pos_box.min.x, pos_box.max.x);
    std::uniform_real_distribution<double> distY(pos_box.min.y, pos_box.max.y);
    std::uniform_real_distribution<double> distZ(pos_box.min.z, pos_box.max.z);

    // velocity uses same box
    std::uniform_real_distribution<double> velX(vel_box.min.x, vel_box.max.x);
    std::uniform_real_distribution<double> velY(vel_box.min.y, vel_box.max.y);
    std::uniform_real_distribution<double> velZ(vel_box.min.z, vel_box.max.z);

    // mass distribution
    std::uniform_real_distribution<double> massDist(massMin, massMax);

    std::vector<gravitationalBody> bodies;
    bodies.reserve(N);

    for (std::size_t i = 0; i < N; ++i) {
        gravitationalBody b;
        b.position = glm::dvec3(distX(rng),
                                distY(rng),
                                distZ(rng));
        b.velocity = glm::dvec3(velX(rng),
                                velY(rng),
                                velZ(rng));
        b.mass     = massDist(rng);
        bodies.push_back(b);
    }

    return bodies;
}

inline std::string get_str_glm_vec3(glm::dvec3 vec, std::string_view name) {
    return std::format("{}: ({}, {}, {})", name, vec.x, vec.y, vec.z);
}

void just_print_glm_vec3(const glm::vec3& vec) {
    std::println("({}, {}, {})", vec.x, vec.y, vec.z);
}

inline void print_glm_vec3(glm::dvec3 vec, std::string_view name) {
    std::cout << get_str_glm_vec3(vec, name) << "\n";
}

inline std::vector<glm::dvec3>
extract_positions(std::vector<gravitationalBody> const& bodies)
{
    std::vector<glm::dvec3> pos;
    pos.reserve(bodies.size());
    for (auto const& b : bodies)
        pos.push_back(b.position);
    return pos;
}

inline void print_positions_fmt(std::vector<glm::dvec3> const& positions)
{
    for (size_t i = 0; i < positions.size(); ++i) {
        auto const& p = positions[i];
        // no extra spaces in the tuple if you want exactly (x,y,z)
        std::cout 
          << std::format("point{}: ({},{},{})\n",
                         i+1,
                         p.x, p.y, p.z);
    }
}

double get_max_dimension(std::vector<gravitationalBody> bodies) {
    std::vector<glm::dvec3> pos_vec = extract_positions(bodies);

    double max_dim = 0.0;

    auto update_max_dim = [&max_dim](double value) {
        if (value > max_dim) {
            max_dim = value;
        }
    };

    for (auto& pos : pos_vec) {
        update_max_dim(pos.x);
        update_max_dim(pos.y);
        update_max_dim(pos.z);
    }

    return max_dim;
}
double calculate_kinetic_energy(std::vector<gravitationalBody>& bodies) {
    double ke {0};
    for (size_t i = 0; i < bodies.size(); i++) {
        gravitationalBody body = bodies[i];
        ke += (0.5l) * body.mass * glm::dot(body.velocity, body.velocity);
    }
    return ke;
}

double calculate_kinetic_energy(simulationFrame& frame) {
    return calculate_kinetic_energy(frame.bodies);
}

double calculate_gpe(std::vector<gravitationalBody>& bodies) {
    double pe {0};
    double grav_G = GRAV_G;
    
    if (bodies.size()<= 1) { return 0.0; }

    for (size_t i = 0; i < bodies.size() - 1; i++) {
        gravitationalBody body1 = bodies[i];
        for (size_t j = i + 1; j < bodies.size(); j++) {
            gravitationalBody body2 = bodies[j];

            glm::dvec3 dr = body1.position - body2.position;
            double seperation = std::sqrt(glm::dot(dr,dr));
            pe -= (grav_G * body1.mass * body2.mass) / seperation;
        }
    }
    return pe;
}

double calculate_gpe(simulationFrame& frame) {
    return calculate_gpe(frame.bodies);
}


inline glm::dvec3 calculate_center_of_mass(std::vector<gravitationalBody> bodies) {
    glm::dvec3 center_of_mass {0};
    double total_mass {0};

    if (bodies.size() <= 1) { return glm::vec3(0.0,0.0,0.0); }

    for (auto& body : bodies) {
        center_of_mass += body.position * body.mass;
        total_mass += body.mass;
    }
    
    center_of_mass /= total_mass;
    return center_of_mass;
}

glm::dvec3 calculate_ang_momentum(simulationFrame& frame, glm::dvec3 origin) {
    // step one calculate center of mass for the system
    // SUM(mass * distance) / SUM(mass)

    glm::dvec3 ang_momentum {};

    for (size_t i = 0; i < frame.bodies.size(); i ++) {
        glm::dvec3 r = frame.bodies[i].position - origin;
        glm::dvec3 p = frame.bodies[i].mass * frame.bodies[i].velocity; // linear momentum
        ang_momentum += glm::cross(r, p);
    }

    return ang_momentum;
}

glm::dvec3 calculate_linear_momentum(simulationFrame& frame) {
    glm::dvec3 linear_momentum {};

    for (size_t i = 0; i < frame.bodies.size(); i ++) {
        linear_momentum += frame.bodies[i].mass * frame.bodies[i].velocity; // linear momentum
    }

    return linear_momentum;
}

template <>
struct std::formatter<glm::dvec3> : std::formatter<std::string> {
    auto format(glm::dvec3 v, format_context& ctx) const {
        return formatter<string>::format(
            std::format("[{:.2e}, {:.2e}, {:.2e}]", v.x, v.y, v.z), ctx
        );
    }
};

void analyse_data_log(std::vector<simulationFrame> log) {
    size_t log_size = log.size();
    glm::vec3 initial_center_of_mass = calculate_center_of_mass(log[0].bodies);

    double kinetic_energy_initial = calculate_kinetic_energy(log[0]);
    double kinetic_energy_final = calculate_kinetic_energy(log[log_size - 1]);

    double potential_energy_initial = calculate_gpe(log[0]);
    double potential_energy_final = calculate_gpe(log[log_size - 1]);


    glm::dvec3 angular_momentum_initial = calculate_ang_momentum(log[0], initial_center_of_mass);
    glm::dvec3 angular_momentum_final = calculate_ang_momentum(log[log_size - 1], initial_center_of_mass);

    glm::dvec3 linear_momentum_initial = calculate_linear_momentum(log[0]);
    glm::dvec3 linear_momentum_final = calculate_linear_momentum(log[log_size - 1]);

    double combined_energy_percentage_difference = (kinetic_energy_final + potential_energy_final) - (kinetic_energy_initial + potential_energy_initial);
    combined_energy_percentage_difference *= (100.0l / (kinetic_energy_initial + potential_energy_initial));

    auto get_vector_magnitude = [](glm::dvec3 v) -> double{
        double x = v.x, y = v.y, z = v.z;
        return std::sqrt(x*x + y*y + z*z);
    };

    glm::dvec3 angular_momentum_difference = angular_momentum_final - angular_momentum_initial;
    double angular_momentum_percentage_difference = 
        100.0l * (get_vector_magnitude(angular_momentum_difference) / get_vector_magnitude(angular_momentum_initial));
    
    glm::dvec3 linear_momentum_difference = linear_momentum_final - linear_momentum_initial;
    double linear_momentum_percentage_difference = 
        100.0l * (get_vector_magnitude(linear_momentum_difference) / get_vector_magnitude(linear_momentum_initial));

    double combined_energy_initial = kinetic_energy_initial + potential_energy_initial;
    double combined_energy_final = kinetic_energy_final + potential_energy_final;


    std::println(
        "{:<25} {:>35} {:>35} {:>18}\n"
        "{}\n"
        "{:<25} {:>35.2e} {:>35.2e}\n"
        "{:<25} {:>35.2e} {:>35.2e}\n"
        "{:<25} {:>35.2e} {:>35.2e} {:>18.2e}%\n"
        "{:<25} {:>35} {:>35} {:>18.2e}%\n"
        "{:<25} {:>35} {:>35} {:>18.2e}%\n",
        "Quantity", "Intial", "Final", "Delta%",
        std::string(25+35+35+18+4, '='),
        "Kinetic Energy", kinetic_energy_initial, kinetic_energy_final, 
        "Potential Energy", potential_energy_initial, potential_energy_final,
        "Combined", combined_energy_initial, combined_energy_final, combined_energy_percentage_difference,
        "Angular Momentum", angular_momentum_initial, angular_momentum_final, angular_momentum_percentage_difference,
        "Linear Momentum", linear_momentum_initial,linear_momentum_final, linear_momentum_percentage_difference);
}
