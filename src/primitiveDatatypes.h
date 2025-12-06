#include <glm/glm.hpp>

#pragma once

struct gravitationalBody {
    double mass; 
    glm::dvec3 position;
    glm::dvec3 velocity;
    double radius;
    double restitution{1.0};
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
//////////////////////////////////
///
