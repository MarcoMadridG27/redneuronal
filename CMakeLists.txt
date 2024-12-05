cmake_minimum_required(VERSION 3.27)
project(redneuronal)

set(CMAKE_CXX_STANDARD 20)
# Incluir directorios de encabezados
include_directories(include)
add_executable(redneuronal src/main.cpp
        src/dataset.cpp
        src/network.cpp
        src/activation.cpp
        src/utils.cpp)
