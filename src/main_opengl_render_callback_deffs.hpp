#include "datatypes.h"
#include "primitiveDatatypes.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GL/glew.h>
#include "main_opengl_render_structs.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <memory>

#ifndef GLWINDOWGLOBSTRUCT
#define GLWINDOWGLOBSTRUCT

#endif // !GLWINDOWGLOBSTRUCT
//
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
bool display_opengl_shader_compilation_error(unsigned int vertexShader); 
bool display_opengl_program_compilation_errordisplay_opengl_program_compilation_error(unsigned int program);


inline void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    auto* g = static_cast<GLWindowGlobals*>(glfwGetWindowUserPointer(window));
    if (yoffset > 0) {
        g->cameraSpeed *= 1.2f;
    } else {
        g->cameraSpeed /= 1.2f;
    }
}

inline void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
    auto* g = static_cast<GLWindowGlobals*>(glfwGetWindowUserPointer(window));
    bool mouse_captured = glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED;
    if (!mouse_captured) { return; }
    float xoffset = xpos - g->lastX;
    float yoffset = g->lastY - ypos; // reversed since y-coordinates range from bottom to top
    g->lastX = xpos;
    g->lastY = ypos;

    const float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    g->yaw   += xoffset;
    g->pitch += yoffset;  

    if (g->pitch > 89.0f) { g->pitch =  89.0f; }

    if (g->pitch < -89.0f) { g->pitch = -89.0f;}
};

inline void processInput(GLFWwindow *window)
{
    auto* g = static_cast<GLWindowGlobals*>(glfwGetWindowUserPointer(window));
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    const float cameraSpeed = g->cameraSpeed * g->deltaTime; // adjust accordingly
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        g->cameraPos += cameraSpeed * g->cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        g->cameraPos -= cameraSpeed * g->cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        g->cameraPos -= glm::normalize(glm::cross(g->cameraFront, g->cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        g->cameraPos += glm::normalize(glm::cross(g->cameraFront, g->cameraUp)) * cameraSpeed;

    uint8_t middle_button_state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE);

    if (middle_button_state == GLFW_PRESS && g->middle_button_state_last_frame == GLFW_RELEASE) {

        if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) 
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        else if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_NORMAL) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);}

        g->middle_button_state_last_frame = true;
    } 
    g->middle_button_state_last_frame = middle_button_state;
}

inline bool display_opengl_shader_compilation_error(unsigned int thisShader) {
    int sucess;
    char infoLog[512];
    glGetShaderiv(thisShader, GL_COMPILE_STATUS, &sucess);

    if (!sucess)
    {
        glGetShaderInfoLog(thisShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
        return true;
    }
    std::cout << "SHADER::COMPILATION::SUCESS\n";
    return false;
}

inline bool display_opengl_program_compilation_error(unsigned int program) {
    int sucess;
    char infoLog[512];
    glGetProgramiv(program, GL_LINK_STATUS, &sucess);
    if (!sucess) {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cout << "ERROR::PROGRAM::COMPILATION_FAILED\n" << infoLog << std::endl;
        return true;
    }
    std::cout << "SHADER_PROGRAM::COMPILATION::SUCESS\n";
    return false;
};

inline void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    auto* g = static_cast<GLWindowGlobals*>(glfwGetWindowUserPointer(window));
    g->width = width;
    g->height= height;
    glViewport(0, 0, width, height);
}


inline double find_average_mass_of_bodies(std::vector<gravitationalBody>& bodies){
    size_t N = bodies.size();
    double total_mass {0};

    for (auto& body : bodies) {
        total_mass += body.mass;
    }

    return (total_mass / N);
}
