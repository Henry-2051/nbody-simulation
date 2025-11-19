// Tell Catch to provide a main() for us:
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
// #include <catch2/catch.hpp>
#include <print>

// Include the headers you want to test:
#include <glm/glm.hpp>
#include "datatypes.h"    // from src/b.hpp
#include "analytics.h"        // from src/body.hpp

// 1) Extract the positions out of your bodies

TEST_CASE("maxBoundingHalfSize on a simple set of points", "[bounding]") {
    // imagine you have a constructor gravitationalBody(glm::vec3 pos, double mass);
    size_t N = 5;
    uint32_t seed = 12345;
    BoundingBox box {{-100,-100,-100}, {100,100,100}};
    std::vector<gravitationalBody> bodies = generateRandomBodies(box, box, N, seed);

    auto pos_vec = extract_positions(bodies);
    double half_size = get_max_dimension(bodies);

    print_positions_fmt(pos_vec);
    std::println("max dimension : {}", half_size);
    // the farthest coordinate in absolute value is 6 (from y = -6)
    // REQUIRE( half_size == Approx(6.0f).epsilon(1e-6f) );
}

// TEST_CASE("maxBoundingHalfSize with symmetric points", "[bounding]") {
//     std::vector<gravitationalBody> bodies = {
//         { glm::vec3{ -1, -1, -1 }, 1.0 },
//         { glm::vec3{  1,  1,  1 }, 1.0 }
//     };
//     auto pos = positions_from_bodies(bodies);
//     REQUIRE( maxBoundingHalfSize(pos) == Approx(1.0f) );
// }

// You can add more TEST_CASE blocks here...
