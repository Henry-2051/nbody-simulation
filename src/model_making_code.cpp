#include "model_making_code.h"


shape_data make_uv_sphere_data(size_t sectorCount = 9, size_t stackCount = 8) {
    std::vector<float> vertices  {};
    std::vector<float> normals   {};
    std::vector<float> texCoords {};

    float radius = 1.0f;
    float x, y, z, xy;
    float nx, ny, nz, lengthInv = 1.0f / radius;
    float s, t;

    float sectorStep = 2.0f * M_PI / sectorCount;
    float stackStep = M_PI / stackCount;
    float sectorAngle, stackAngle;

    for (size_t i = 0; i <= stackCount; ++i) {
        stackAngle = M_PI / 2 - i * stackStep;

        xy = radius * std::cosf(stackAngle);
        z = radius * std::sinf(stackAngle);

        for (size_t j {0}; j <= sectorCount; ++j) {
            sectorAngle = j * sectorStep;           // starting from 0 to 2pi

            // vertex position (x, y, z)
            x = xy * std::cosf(sectorAngle);             // r * cos(u) * cos(v)
            y = xy * std::sinf(sectorAngle);             // r * cos(u) * sin(v)
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);

            // normalized vertex normal (nx, ny, nz)
            nx = x * lengthInv;
            ny = y * lengthInv;
            nz = z * lengthInv;
            normals.push_back(nx);
            normals.push_back(ny);
            normals.push_back(nz);

            // vertex tex coord (s, t) range between [0, 1]
            s = (float)j / sectorCount;
            t = (float)i / stackCount;
            texCoords.push_back(s);
            texCoords.push_back(t);
        }
    }
    return {vertices, normals, texCoords}; // shape struct
}

shape_indices make_uv_sphere_indices(size_t sectorCount = 8, size_t stackCount = 8) {
    // generate CCW index list of sphere triangles
    // k1--k1+1
    // |  / |
    // | /  |
    // k2--k2+1
    std::vector<uint> indices;
    std::vector<uint> lineIndices;
    uint k1, k2;
    for(size_t i {0}; i < stackCount; ++i)
    {
        k1 = i * (sectorCount + 1);     // beginning of current stack
        k2 = k1 + sectorCount + 1;      // beginning of next stack

        for(size_t j {0}; j < sectorCount; ++j, ++k1, ++k2)
        {
            // 2 triangles per sector excluding first and last stacks
            // k1 => k2 => k1+1
            if(i != 0)
            {
                indices.push_back(k1);
                indices.push_back(k2);
                indices.push_back(k1 + 1);
            }

            // k1+1 => k2 => k2+1
            if(i != (stackCount-1))
            {
                indices.push_back(k1 + 1);
                indices.push_back(k2);
                indices.push_back(k2 + 1);
            }

            // store indices for lines
            // vertical lines for all stacks, k1 => k2
            lineIndices.push_back(k1);
            lineIndices.push_back(k2);
            if(i != 0)  // horizontal lines except 1st stack, k1 => k+1
            {
                lineIndices.push_back(k1);
                lineIndices.push_back(k1 + 1);
            }
        }
    }
    return {indices, lineIndices};
}
