#pragma once

#include <glm/glm.hpp>

struct gravitationalBody {
    double mass; 
    glm::dvec3 position;
    glm::dvec3 velocity;
    double radius;
    double restitution{1.0};
};


struct simulationFrame {
    std::vector<gravitationalBody> bodies;
    double time;
    // by default we are going to assume that every element of the array 
    // contains a valid gravitational body datapoint, if the situation arrises where we want to test
    // something weird like add another gravitational body mid simulation we will have to take care 
    // of this.
};

struct BoundingBox {
    glm::dvec3 min;
    glm::dvec3 max;
};

struct SystemMetrics {
    double combined_energy;
    size_t system_time;
};

//////////////////////////////
// opengl specific 
struct vertex_f {
    float x, y, z;
};

struct color_f {
    float r, g, b;
};

struct colored_vertex_f {
    vertex_f v;
    color_f  c;
};
////////////////////////////////
