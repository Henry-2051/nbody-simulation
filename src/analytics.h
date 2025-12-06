#include <array>
#include <cmath>
#include <format>

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
#include "preprocessorDefinitions.h"

#pragma once


// A gravitational body.
// double mass  8 bytes
// glm::dvec3 position 24 bytes
// glm::dvec3 velocity 24 bytes

inline std::ostream& operator<<(std::ostream& os, glm::dvec3 const& v)
{
    return os << '(' << v.x << ", " << v.y << ", " << v.z << ')';
}

// using collision_candidate_function_sig = std::function<void(size_t, const std::vector<gravitationalBody>&, std::vector<std::vector<size_t>>&)>;




inline std::string get_str_glm_vec3(glm::dvec3 vec, std::string_view name) {
    return std::format("{}: ({}, {}, {})", name, vec.x, vec.y, vec.z);
}

inline void just_print_glm_vec3(const glm::vec3& vec) {
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
