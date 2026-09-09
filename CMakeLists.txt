cmake_minimum_required(VERSION 3.20)

project(giao_integrals VERSION 0.5.0 LANGUAGES CXX)

include(GNUInstallDirs)

option(GIAO_BUILD_PYTHON "Build the pybind11 extension" ON)
option(GIAO_BUILD_TESTS "Build the C++ unit tests" ON)
option(GIAO_BUILD_BENCHMARKS "Build the overlap benchmark" OFF)
option(GIAO_ENABLE_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)

add_library(giao_integrals_core
    src/boys.cpp
    src/eri.cpp
    src/hermite.cpp
    src/nuclear.cpp
    src/overlap.cpp
    src/property.cpp
    src/types.cpp
)
add_library(giao_integrals::core ALIAS giao_integrals_core)
target_compile_features(giao_integrals_core PUBLIC cxx_std_20)
target_include_directories(giao_integrals_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_target_properties(giao_integrals_core PROPERTIES
    EXPORT_NAME core
    POSITION_INDEPENDENT_CODE ON
)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(giao_integrals_core PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
endif()

if(GIAO_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(giao_integrals_core PUBLIC
        -fsanitize=address,undefined -fno-omit-frame-pointer
    )
    target_link_options(giao_integrals_core PUBLIC
        -fsanitize=address,undefined
    )
endif()

if(GIAO_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_giao_integrals src/bindings.cpp)
    target_link_libraries(_giao_integrals PRIVATE giao_integrals::core)
    target_compile_features(_giao_integrals PRIVATE cxx_std_20)
    install(TARGETS _giao_integrals LIBRARY DESTINATION giao_integrals)
endif()

include(CTest)
if(BUILD_TESTING AND GIAO_BUILD_TESTS)
    add_executable(giao_cpp_tests tests/cpp/test_overlap.cpp)
    target_link_libraries(giao_cpp_tests PRIVATE giao_integrals::core)
    add_test(NAME giao_cpp_tests COMMAND giao_cpp_tests)
endif()

if(GIAO_BUILD_BENCHMARKS)
    add_executable(giao_overlap_benchmark benchmarks/overlap_benchmark.cpp)
    target_link_libraries(giao_overlap_benchmark PRIVATE giao_integrals::core)
endif()

install(TARGETS giao_integrals_core
    EXPORT giao_integralsTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT giao_integralsTargets
    FILE giao_integralsTargets.cmake
    NAMESPACE giao_integrals::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/giao_integrals
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/giao_integralsConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/giao_integralsConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/giao_integrals
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/giao_integralsConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/giao_integralsConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/giao_integralsConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/giao_integrals
)
