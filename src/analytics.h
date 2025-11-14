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
#include <Eigen/Eigen>

#define GRAV_G 6.67408e-11
#define MASS_EARTH 5.9722e24
#define MASS_MOON 7.348e22
#define DIST_EARTH_MOON 3.84e8
#define MOON_ANG_FREQ 2.66e-6
#define MOON_EARTH_VELOCITY (MOON_ANG_FREQ * DIST_EARTH_MOON)

struct gravitationalBody {
    double mass;
    glm::dvec3 position;
    glm::dvec3 velocity;
};


template <size_t N>
struct simulationFrame {
    std::array<gravitationalBody, N> bodies;
    float time;
    size_t bodies_contained{N}; 
    // by default we are going to assume that every element of the array 
    // contains a valid gravitational body datapoint, if the situation arrises where we want to test
    // something weird like add another gravitational body mid simulation we will have to take care 
    // of this.
};

inline std::string get_str_glm_vec3(glm::dvec3 vec, std::string_view name) {
    return std::format("{}: ({}, {}, {})", name, vec.x, vec.y, vec.z);
}

inline void print_glm_vec3(glm::dvec3 vec, std::string_view name) {
    std::cout << get_str_glm_vec3(vec, name) << "\n";
}

template <size_t N>
float calculate_kinetic_energy(simulationFrame<N>& frame) {
    double ke {0};
    for (size_t i = 0; i < frame.bodies_contained; i++) {
        gravitationalBody body = frame.bodies[i];
        ke += (0.5l) * body.mass * glm::dot(body.velocity, body.velocity);
    }
    return ke;
}

template <size_t N>
float calculate_gpe(simulationFrame<N>& frame) {
    double pe {0};
    double grav_G = GRAV_G;
    
    if (frame.bodies_contained <= 1) { return 0.0; }

    for (size_t i = 0; i < frame.bodies_contained - 1; i++) {
        gravitationalBody body1 = frame.bodies[i];
        for (size_t j = i + 1; j < frame.bodies_contained; j++) {
            gravitationalBody body2 = frame.bodies[j];

            glm::dvec3 dr = body1.position - body2.position;
            double seperation = std::sqrt(glm::dot(dr,dr));
            pe -= (grav_G * body1.mass * body2.mass) / seperation;
        }
    }

    return pe;
}

template <size_t N>
glm::dvec3 calculate_center_of_mass(std::array<gravitationalBody, N>& bodies, size_t num_bodies) {
    glm::dvec3 center_of_mass {0};
    double total_mass {0};

    if (num_bodies <= 1) { return glm::dvec3(0.0,0.0,0.0); }

    for (size_t i = 0; i < num_bodies; i ++) {
        gravitationalBody body = bodies[i];
        center_of_mass += body.position * body.mass;
        total_mass += body.mass;
    }

    center_of_mass /= total_mass;

    return center_of_mass;
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

template <size_t N>
glm::dvec3 calculate_ang_momentum(simulationFrame<N>& frame, glm::dvec3 origin) {
    // step one calculate center of mass for the system
    // SUM(mass * distance) / SUM(mass)

    glm::dvec3 ang_momentum {};

    for (size_t i = 0; i < frame.bodies_contained; i ++) {
        glm::dvec3 r = frame.bodies[i].position - origin;
        glm::dvec3 p = frame.bodies[i].mass * frame.bodies[i].velocity; // linear momentum
        ang_momentum += glm::cross(r, p);
    }

    return ang_momentum;
}

template <size_t N>
glm::dvec3 calculate_linear_momentum(simulationFrame<N>& frame) {
    glm::dvec3 linear_momentum {};

    for (size_t i = 0; i < frame.bodies_contained; i ++) {
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

template <size_t N>
void analyse_data_log(std::vector<simulationFrame<N>> log) {
    size_t log_size = log.size();
    glm::vec3 initial_center_of_mass = calculate_center_of_mass(log[0].bodies, log[0].bodies_contained);

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
