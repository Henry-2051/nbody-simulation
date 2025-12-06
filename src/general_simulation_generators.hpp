#include "primitiveDatatypes.h"
#include "preprocessorDefinitions.h"
#include <random>

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

