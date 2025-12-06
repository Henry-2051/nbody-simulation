#include <vector>
#include "primitiveDatatypes.h"
#include "integration_schemes.h"
#include "general_simulation_generators.hpp"

inline std::vector<gravitationalBody> generate_thousand_random_bodies() {
    BoundingBox pos_box {{-1000,-1000,-1000}, {1000,1000,1000}};
    pos_box.min *= 150.0;
    pos_box.max *= 150.0;

    BoundingBox vel_box {{0,0,0}, {1,1,1}};
    vel_box.min *= 1.0;
    vel_box.max *= 10.0;

    const std::size_t num_points      = 400;
    uint32_t    seed   = 123456;    
    double      mMin   = 1e10;
    double      mMax   = 1e11;

    std::vector<gravitationalBody> bodies = generateRandomBodies(pos_box, vel_box, num_points, seed, mMin, mMax);

    std::vector<double> densities = std::vector<double>(1000); // kg m^-3
    std::fill(densities.begin(), densities.end(), 2000.0);
    calculate_spherical_radius(bodies, densities);

    for (int i = 0; i < bodies.size(); i++) {
        bodies[i].restitution = 0.6;
    }

    return bodies;
}

inline std::function<std::vector<gravitationalBody>()> generate_three_body_generator(size_t seed) {
    return [seed](){
        BoundingBox pos_box {{-1000,-1000,-1000}, {1000,1000,1000}};
        pos_box.min *= 1.0;
        pos_box.max *= 1.0;

        BoundingBox vel_box {{-1,-1,-1}, {1,1,1}};
        vel_box.min *= 0.01;
        vel_box.max *= 0.01;

        const std::size_t num_points   = 3;
        double      mMin   = 1e10;
        double      mMax   = 1e10;

        std::vector<gravitationalBody> bodies = generateRandomBodies(pos_box, vel_box, num_points, seed, mMin, mMax);

        std::vector<double> densities = {2000.0, 2000.0, 2000.0}; // kg m^-3
        bodies[0].restitution = 1.0;
        bodies[1].restitution = 1.0;
        bodies[2].restitution = 1.0;
        
        calculate_spherical_radius(bodies, densities);

        return bodies;
    };
}

inline std::vector<gravitationalBody> earth_moon_bodies() {
    return {
        {MASS_EARTH, {0, 0,0},{0, 0,0}},
        {MASS_MOON, {0,DIST_EARTH_MOON,0},{MOON_EARTH_VELOCITY,0,0}},
    };
}
