#include <glm/glm.hpp>
#include <iostream>
#include <stdexcept>

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
    float& operator[] (size_t index);
};

struct color_f {
    float r, g, b;
    float& operator[](std::size_t index);
};

struct colored_vertex_f {
    vertex_f v;
    color_f  c;
};

//////////////////////////////////
///

/////////////////////////////////////
///model code
struct shape_data {
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> texCoords;
};

struct shape_indices {
    std::vector<uint> indices;
    std::vector<uint> lineIndices;
};

struct shape {
    shape_data data;
    shape_indices indices;
};

struct fd_shape {
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> texCoords;
    std::vector<uint> indices;
    std::vector<uint> lineIndices;
};
/////////////////////////////////////


inline void printS(vertex_f v) {
    std::cout << std::format("vertex_f       : {}, {}, {}\n",v.x, v.y, v.z);
}

inline void printS(glm::vec3 v3) {
    std::cout << std::format("glm vec3       : {}, {}, {}\n", v3.x, v3.y, v3.z);
} 
