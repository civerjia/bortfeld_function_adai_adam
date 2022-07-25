cmake_minimum_required(VERSION 3.4...3.18)
project(Bortfeld LANGUAGES CXX)

add_subdirectory(pybind11)
pybind11_add_module(Bortfeld src/BortfeldFunction_py.cpp src/bp.cpp src/parabolic_cylinder_function.cpp src/parabolic_cylinder_function.h src/bp.h)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(Bortfeld PUBLIC OpenMP::OpenMP_CXX)
endif()

find_package (Python COMPONENTS Interpreter Development)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
#target_compile_definitions(Bortfeld PRIVATE VERSION_INFO=${v1.0})
