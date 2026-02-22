#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <memory>

#include "datatypes.h"
#include "analytics.h"
#include "main_opengl_render_callback_deffs.hpp"
#include "main_opengl_render_structs.hpp"
#include "model_making_code.h"
#include "primitiveDatatypes.h"
#include "opengl_imgui_menu.hpp"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
bool display_opengl_shader_compilation_error(unsigned int vertexShader); 
bool display_opengl_program_compilation_error(unsigned int program);


inline int openglDisplay(simulation_description sim_desc, GLWindowGlobals globals, SimulationGLobals sim_globals) {
    GLuint spherePosVBO, drawingAsteroidsVAO, EBO, pointcloud_vert_shader, fragment_shader, sphereMesh_vert_shader;
    GLuint uvSphereGeometryVBO, uvSphereEboTrianges, uvSphereEboLines;
    auto *gs = &sim_globals;
    auto *g = static_cast<GLWindowGlobals*>(&globals);

    gs->desc = sim_desc;

    {
        shape my_sphere {};
        my_sphere.data = make_uv_sphere_data(8, 8);
        my_sphere.indices = make_uv_sphere_indices(8, 8);

        g->uv_sphere.vertices = my_sphere.data.vertices;
        g->uv_sphere.normals = my_sphere.data.normals;
        g->uv_sphere.texCoords = my_sphere.data.texCoords;
        g->uv_sphere.indices = my_sphere.indices.indices;
        g->uv_sphere.lineIndices = my_sphere.indices.lineIndices;
    }

    const char * vertex_shader_source = R"glsl(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        // layout (location = 1) in vec3 aColor;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        out vec3 vertColor;

        void main()
        {
            gl_Position = projection * view * model * vec4(aPos, 1.0f);
            vertColor = vec3(1.0,1.0,1.0);
        }
    )glsl";

    const char * vertex_shader_source_meshSphere = R"glsl(
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec3 aOffset;

        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;

        uniform vec3 iColor;

        out vec3 vertColor;

        void main()
        {
            gl_Position = projection * view * model * vec4(aPos + aOffset, 1.0f);
            vertColor = iColor;
        }
    )glsl";

    const char * fragment_shader_source = R"glsl(
    #version 330 core
    out vec4 FragColor;

    in vec3 vertColor;

    void main()
    {
        FragColor = vec4(vertColor, 1.0f);
    }
    )glsl";

    std::vector<gravitationalBody> bodies = sim_desc.gen_bodies();
    // std::vector<gravitationalBody> bodies = generate_thousand_random_bodies();

    std::vector<glm::vec3> float_body_positions (bodies.size());

    // helper function to ensure float_bodies_positions mirrors the positions 
    // within bodies but using vec3 rather than dvec3 for opengl
    auto copyover_float_position_from_double = [&bodies, &float_body_positions](){
        if (bodies.size() != float_body_positions.size()) {
            float_body_positions.resize(bodies.size());
        }
        for (int i = 0; i < bodies.size(); i ++) {
            float_body_positions[i] = bodies[i].position;
        }
    };

    double max_dimension = get_max_dimension(bodies);
    float draw_scale_factor = 1000.0 / max_dimension;
    
    // std::println("scale factor calculated as : {}", draw_scale_factor);

    auto scale_float_positions = [&float_body_positions, draw_scale_factor]() {
        for (auto& pos : float_body_positions) {
            pos *= draw_scale_factor;
        }
    };


    copyover_float_position_from_double(); // lambda call
    scale_float_positions(); // 

    // TODO : should put system metrics like this into their own struct or container 
    double combined_energy_last = calculate_gpe(bodies) + calculate_kinetic_energy(bodies);

    // this is the log that is generated 
    g->metric_log = std::make_unique<SystemMetrics[]>(g->metric_log_length);
    g->metric_log[0] = SystemMetrics(combined_energy_last, 0);

    // now atomic 
    auto calculate_system_metrics = [&bodies, gs, &combined_energy_last]() {
            gs->combined_energy_current = calculate_kinetic_energy(bodies) + calculate_gpe(bodies);
            gs->perc_energy_divergence = 100.0 * (gs->combined_energy_current - combined_energy_last) / combined_energy_last;
    };

    // for (auto& body : bodies) {
    //     std::println("Body radius : {}", body.radius);
    //     std::println("Body mass : {}", body.mass);
    // }

    // size_t max_points = bodies.size() * 2;

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(g->width, g->height, "nbody", NULL, NULL);
    glfwSetWindowUserPointer(window, &globals);

    if (window == nullptr) {
        std::cerr << "Error, unable to initialise glfw window :(";
    }

    glfwMakeContextCurrent(window);
        
    glEnable(GL_DEPTH_TEST);  
    glewInit();

    glViewport(0, 0, 800, 600); // this is the size of the rendering window, 
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);  
    glfwSetScrollCallback(window, scroll_callback); 
    glfwSetCursorPosCallback(window, mouse_callback);  

    glGenVertexArrays(1, &drawingAsteroidsVAO);
    glGenBuffers(1, &spherePosVBO);
    glGenBuffers(1, &EBO);
    glGenBuffers(1, &uvSphereGeometryVBO);
    glGenBuffers(1, &uvSphereEboTrianges);
    glGenBuffers(1, &uvSphereEboLines);
    // bind the vertex array object first, then bind and se the vertex buffer(s), and then configure vertex attributes
    glBindVertexArray(drawingAsteroidsVAO);

    glBindBuffer(GL_ARRAY_BUFFER, spherePosVBO);
    // function designed to copy user defined data into the currently bound buffer
    glBufferData(GL_ARRAY_BUFFER, bodies.size() * sizeof(glm::vec3), nullptr, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, uvSphereGeometryVBO);
    glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(float) * g->uv_sphere.vertices.size(), g->uv_sphere.vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, uvSphereEboTrianges);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 3 * sizeof(uint) * g->uv_sphere.indices.size(), g->uv_sphere.indices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, uvSphereEboLines);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * sizeof(uint) * g->uv_sphere.lineIndices.size(), g->uv_sphere.lineIndices.data(), GL_STATIC_DRAW);


    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, spherePosVBO);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, (void*)0);
    glVertexAttribDivisor(0,0);

    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, uvSphereGeometryVBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glVertexAttribDivisor(1,1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, uvSphereEboTrianges);
    glBindVertexArray(0);


    // atribute pointers and data layout configuration
    // param 1 : location of the configured attibute (look at vertex shader)
    // param 2 : size of the vertex attribute, composed of 3 values
    // param 3 : the type of the data, in opengl a vec3 is composed of floats
    // param 4 : this is whether we want the data normalised, this would be true for integer data types
    // param 5 : this is known as the stride length, it tells us the length between consecutive values
    // param 6 : type (void *) tells us the offset of where the position data begins in the buffer.


    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_source, NULL);
    display_opengl_shader_compilation_error(fragment_shader);
    std::cout << "fragment shader finished\n===============================\n";

    sphereMesh_vert_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(sphereMesh_vert_shader, 1, &vertex_shader_source_meshSphere, NULL);
    glCompileShader(sphereMesh_vert_shader);

    display_opengl_shader_compilation_error(sphereMesh_vert_shader);
    std::cout << "spheremesh shader finished\n===============================\n";



    g->spheremesh_shader_program = glCreateProgram();
    glAttachShader(g->spheremesh_shader_program, sphereMesh_vert_shader);
    glAttachShader(g->spheremesh_shader_program, fragment_shader);
    glLinkProgram(g->spheremesh_shader_program);


    display_opengl_program_compilation_error(g->spheremesh_shader_program);
    std::cout << "spheremesh program finished\n";

    glDeleteShader(sphereMesh_vert_shader);
    glDeleteShader(pointcloud_vert_shader);
    glDeleteShader(fragment_shader);

    glm::mat4 projection, view, model;
    // this is the only place where we are conducting pointer shenanigans with this variable, its 
    // good practice to free the resource before assigning a new resource, preventing a memory leak
    
    delete gs->inte;
    gs->inte = new integrator(sim_desc.integrator_type, sim_desc.collision_resolution_type, sim_desc.acceleration_function, sim_desc.integrator_step_size_hint);
    
    // auto last_frame = std::chrono::high_resolution_clock::now();
    gs->current_simulation_time = sim_desc.start;
    double next_target_time = sim_desc.start + sim_desc.simulation_step_size;

    float time_accumulator = 0;
    size_t frame_count = 0;

    // IMGUI SETUP
    // Create window with graphics context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    ImGui::StyleColorsDark();


    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    while((!glfwWindowShouldClose(window)) &&
        gs->current_simulation_time < sim_desc.end) 
    {
        if (!g->paused) frame_count ++;

        float current_time = glfwGetTime();
        g->deltaTime = current_time - g->lastFrame;
        g->lastFrame = current_time;
        time_accumulator += g->deltaTime;

        // calculate_gpe is a relatively expensive call of order n^2, comparable to timestepping the system
        if ((!g->paused) && g->system_analytics_window) {
            calculate_system_metrics();
        }

        display_imgui_menu(g, gs, io);

        if (!g->paused) {
            combined_energy_last = gs->combined_energy_current;

            gs->current_simulation_time = gs->inte->integrate(bodies, gs->current_simulation_time, next_target_time);
            next_target_time += gs->desc.simulation_step_size;
            copyover_float_position_from_double();
            scale_float_positions();
        }

        // for (auto& pos : float_body_positions) {
        //     std::println("({}, {}, {})", pos.x, pos.y, pos.z);
        // }

        // just_print_glm_vec3(g->cameraPos);
        // std::println();

        processInput(window);
        glm::vec3 direction= glm::vec3(
            cos(glm::radians(g->yaw)) * cos(glm::radians(g->pitch)),
            sin(glm::radians(g->pitch)),
            sin(glm::radians(g->yaw)) * cos(glm::radians(g->pitch))
        );
 
        model = glm::mat4(1.0f);

        projection = glm::perspective(glm::radians(g->fov), 
                                      (float)g->width / (float)g->height, 
                                      0.1f, g->far_plane_view_distance);  

        g->cameraFront = glm::normalize(direction);

        view = glm::lookAt(g->cameraPos, 
                           g->cameraPos + g->cameraFront, 
                           g->cameraUp);

        GLuint shader_program;
        shader_program = g->spheremesh_shader_program;

        glUniformMatrix4fv(glGetUniformLocation(shader_program, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shader_program, "view"), 1, GL_FALSE, glm::value_ptr(view));


        glClearColor(g->clear_color.x * g->clear_color.w, g->clear_color.y * g->clear_color.w, g->clear_color.z * g->clear_color.w, g->clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


        glUseProgram(shader_program);

        glBindVertexArray(drawingAsteroidsVAO);
        GLint uColor = glGetUniformLocation(shader_program, "iColor");
        glUniform3fv(uColor, 1, glm::value_ptr(g->sphere_color));

        glBindBuffer(GL_ARRAY_BUFFER, spherePosVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, bodies.size() * sizeof(glm::vec3), float_body_positions.data());

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, uvSphereEboTrianges);
        glDrawElementsInstanced(GL_TRIANGLES, 3 * g->uv_sphere.indices.size(), GL_UNSIGNED_INT, (void*)0, bodies.size());

        glUniform3fv(uColor, 1, glm::value_ptr(g->line_color));
        glDrawElementsInstanced(GL_LINES, 2 * g-> uv_sphere.lineIndices.size(), GL_UNSIGNED_INT, 0, bodies.size());

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glBindBuffer(GL_ARRAY_BUFFER, 0); 
        glfwSwapBuffers(window);
        glfwPollEvents();



    }


    glfwTerminate();

    return 0;
}
