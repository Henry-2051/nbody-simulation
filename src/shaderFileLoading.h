#ifndef SHADER_H
#define SHADER_H
  
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

inline std::string loadFileToString(const char* filepath) {
    std::string file_text;
    std::ifstream file;

    file.exceptions (std::ifstream::failbit | std::ifstream::badbit);
    try
    {
        file.open(filepath);
        std::stringstream file_string_stream;

        file_string_stream << file.rdbuf();

        file.close();

        file_text = file_string_stream.str();
    }
    catch(const std::exception& e)
    {
        std::cerr << "Failed to open: " << filepath << "\n" << e.what() << '\n';
    }
    return file_text;
}

#endif
