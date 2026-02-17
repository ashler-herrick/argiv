cmake_minimum_required(VERSION 3.20)
project(argiv LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QuantLib CONFIG REQUIRED)
find_package(Arrow CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

pybind11_add_module(_core
    src/argiv/core.cpp
    src/argiv/arrow_interop.cpp
    src/argiv/compute.cpp
    src/argiv/surface.cpp
    src/bindings/python_module.cpp
)

target_include_directories(_core PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(_core PRIVATE QuantLib::QuantLib Arrow::arrow_static)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS _core LIBRARY DESTINATION .)

# Install Python source files
install(FILES
    argiv/__init__.py
    argiv/helpers.py
    DESTINATION .
)

# Install Python stub files and py.typed marker for type hints (PEP 561)
install(FILES
    argiv/__init__.pyi
    argiv/_core.pyi
    argiv/py.typed
    DESTINATION .
)
