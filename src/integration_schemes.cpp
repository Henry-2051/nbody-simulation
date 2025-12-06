#include "integration_schemes.h"
#include "preprocessorDefinitions.h"
#include <glm/fwd.hpp>

void _sum2_vec_dvec3(std::vector<glm::dvec3>& vec1, const std::vector<glm::dvec3>& vec2) {
    for (size_t i = 0; i < vec1.size(); i++) {
        vec1[i] += vec2[i]; 
    }
}

std::vector<glm::dvec3>& operator+=(std::vector<glm::dvec3>& vec1, const std::vector<glm::dvec3>& vec2) {
    _sum2_vec_dvec3(vec1, vec2);
    return vec1;
}

void _scalarMul2_vec_dvec3(std::vector<glm::dvec3>& vec1, double scalar) {
    for (size_t i = 0; i < vec1.size(); i++) {
        vec1[i] *= scalar;
    }
}

std::vector<glm::dvec3>& operator*=(std::vector<glm::dvec3>& vec1, double scalar) {
    _scalarMul2_vec_dvec3(vec1, scalar);
    return vec1;
}

void calculate_gravitational_acceleration(const std::vector<gravitationalBody> &bodies, std::vector<glm::dvec3>& acceleration) {
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

struct barnesHuttNode{
    double mass;
    glm::dvec3 position;
    glm::dvec3 center_of_mass;
    size_t chiledren[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    size_t bodies_contained = 0;
};

using bh8Tree = std::vector<barnesHuttNode>;

void regenerate_barnes_hutt_tree(const std::vector<gravitationalBody> &bodies, bh8Tree &bhtree) {
    bhtree.clear();
    bhtree.push_back({0.0, {0.0, 0.0, 0.0}});


    for (int i = 0; i < bodies.size(); i++){
        bool found_leaf = false;
        size_t current_node = 0;
        while (!found_leaf) {
            double mNode = bhtree[current_node].mass;
            glm::dvec3 pNode = bhtree[current_node].position;
            pNode = (pNode * mNode + bodies[i].position * bodies[i].mass) / (mNode + bodies[i].mass);
            mNode += bodies[i].mass;

            bhtree[current_node].position = pNode;
            bhtree[current_node].mass = mNode;
            bhtree[current_node].bodies_contained ++;

            // calcualte the new nodes octant


        }
    }
}

void barnes_hutt_calculate_gravitational_acceleraiton(const std::vector<gravitationalBody> &bodies, std::vector<glm::dvec3>& acceleration) {
    
}

void timestep_euler(std::vector<gravitationalBody>& bodies, std::vector<glm::dvec3>& acceleration_junk, double step_size, accel_func_signiture acc_func) {
    acceleration_junk.resize(bodies.size());

    acc_func(bodies, acceleration_junk);
    std::vector<glm::dvec3> acceleration = std::move(acceleration_junk);

    for (size_t i = 0; i < bodies.size(); i++) {
        bodies[i].position += bodies[i].velocity * step_size; 
        bodies[i].velocity += acceleration[i] * step_size;
    }

    acceleration_junk = std::move(acceleration);
}
void timestep_RK2(
    std::vector<gravitationalBody>& bodies, 
    std::vector<gravitationalBody>& body_copy, 
    std::vector<glm::dvec3>& acceleration_junk, 
    double step_size, 
    accel_func_signiture acc_func,
    forward_euler_function_signiture_interface forward_euler) 
{
    body_copy.resize(bodies.size());
    acceleration_junk.resize(bodies.size());

    std::copy(bodies.begin(), bodies.end(), body_copy.begin());
    forward_euler(body_copy, acceleration_junk, step_size / 2.0l, acc_func);

    acc_func(body_copy, acceleration_junk);
    std::vector<glm::dvec3> acceleration = std::move(acceleration_junk);

    for (size_t i = 0; i < bodies.size(); i++) {
        bodies[i].position += body_copy[i].velocity * step_size;
        bodies[i].velocity += acceleration[i] * step_size;
    }

    acceleration_junk = std::move(acceleration);
}

void timestep_RK4(
    std::vector<gravitationalBody>& bodies, std::array<std::vector<gravitationalBody>, 3>& body_copies,
    std::array<std::vector<glm::dvec3>, 4>& acceleration_junk,
    double step_size,
    accel_func_signiture acc_func
)
{
    for (int i = 0; i < 4; i++) {
        acceleration_junk[i].resize(bodies.size());
        if (i != 3) {
            body_copies[i].resize(bodies.size());
            std::copy(bodies.begin(), bodies.end(), body_copies[i].begin());
        }
    }
    accel_func_signiture dummy_acceleration_function = [](const std::vector<gravitationalBody>& bodies, std::vector<glm::dvec3>& acceleration) {};

    auto& x2 = body_copies[0];
    auto& x3 = body_copies[1];
    auto& x4 = body_copies[2];

    auto& a1 = acceleration_junk[0]; 
    auto& a2 = acceleration_junk[1];
    auto& a3 = acceleration_junk[2];
    auto& a4 = acceleration_junk[3];

    // were actually doing 2 seperate forward euler steps for each stage of the runge kutta

    acc_func(bodies, a1);
    for (size_t i = 0; i < bodies.size(); i ++) {
        x2[i].position += bodies[i].velocity * (step_size / 2.0);
        x2[i].velocity += a1[i] * (step_size / 2.0);
    }
    
    acc_func(x2, a2);
    for (size_t i = 0; i < bodies.size(); i ++) {
        x3[i].position += x2[i].velocity * (step_size / 2.0);
        x3[i].velocity += a2[i] * (step_size / 2.0);
    }

    acc_func(x3, a3);
    for (size_t i = 0; i < bodies.size(); i ++) {
        x4[i].position += x3[i].velocity * (step_size);
        x4[i].velocity += a3[i] * (step_size);
    }

    acc_func(x4, a4);
    for (size_t i = 0; i < bodies.size(); i++) {
        bodies[i].position += (step_size / 6.0) * (bodies[i].velocity + 2.0 * x2[i].velocity + 2.0 * x3[i].velocity + x4[i].velocity);
        bodies[i].velocity += (step_size / 6.0) * (a1[i] + 2.0 * a2[i] + 2.0 * a3[i] + a4[i]);
    }
}

// COLLISION RESOLUTION CODE

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
                // std::cout << std::format("inside!!!{} \n", bodies[j].position.x);
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

