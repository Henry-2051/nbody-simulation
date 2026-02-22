#include "datatypes.h"
#include "main_opengl_render_structs.hpp"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

inline void display_imgui_menu(GLWindowGlobals* g, SimulationGLobals* gs, ImGuiIO& io) {


    // IMGUI DISPLAY
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    {
        ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.
        //
        // ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
        ImGui::Checkbox("System analytics", &g->system_analytics_window);      // Edit bools storing our window open/close state
        ImGui::Checkbox("System control", &g->system_control);

        ImGui::Checkbox("paused", &g->paused);
        ImGui::ColorEdit3("clear color", (float*)&g->clear_color); // Edit 3 floats representing a color
        ImGui::SliderFloat("point size", &g->point_draw_size, 1.0f, 20.0f);


        ImGui::Text("Simulation world time: %.0f seconds\nSimulation progress: %.1f%%", gs->current_simulation_time, (((gs->current_simulation_time - gs->desc.start) * 100.0) / (gs->desc.end - gs->desc.start)));
        ImGui::Text("Simulation frame: %zu", g->frame_count);

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        ImGui::End();
    }

    if (g->system_analytics_window) {
        ImGui::Begin("System analtics", &g->system_analytics_window);

        ImGui::Text("%s", std::format("current energy of the systme  : {:.0f} Joules", gs->combined_energy_current).c_str());
        ImGui::Text("%s", std::format("percentage energy change from last frame : {:.2e}%%", gs->perc_energy_divergence).c_str());
        double perc_energy_change_from_start = 100.0 * ((gs->combined_energy_current - g->metric_log[0].combined_energy) / g->metric_log[0].combined_energy);
        ImGui::Text("%s", std::format("percentage energy change from start : {:.2e}%%", perc_energy_change_from_start).c_str());
        ImGui::End();
    }

    if (g->system_control) {
        ImGui::Begin("System control", &g->system_control);
        double min = gs->desc.simulation_step_size * 0.1, max = gs->desc.simulation_step_size * 10.0;
        ImGui::SliderScalar("simulation time per frame", ImGuiDataType_Double, 
                            &gs->desc.simulation_step_size, &min, &max);

        min = gs->desc.integrator_step_size_hint* 0.1, max = gs->desc.integrator_step_size_hint * 10.0;
        ImGui::SliderScalar("integrator step size", ImGuiDataType_Double, &gs->inte->step_size, &min, &max);
        ImGui::End();
    }
}
