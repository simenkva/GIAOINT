# AUTO detects OpenMP without making a source installation depend on it.
set(GIAO_ENABLE_OPENMP "AUTO" CACHE STRING "OpenMP support: AUTO, ON, or OFF")
set_property(CACHE GIAO_ENABLE_OPENMP PROPERTY STRINGS AUTO ON OFF)
string(TOUPPER "${GIAO_ENABLE_OPENMP}" GIAO_ENABLE_OPENMP)
if(NOT GIAO_ENABLE_OPENMP MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "GIAO_ENABLE_OPENMP must be AUTO, ON, or OFF")
endif()
set(GIAO_OPENMP_ENABLED OFF)
unset(_giao_openmp_blocked)

if(NOT GIAO_ENABLE_OPENMP STREQUAL "OFF")
    # AppleClang does not ship a runtime. Select a matching header/library pair
    # before FindOpenMP can choose Homebrew instead of NumPy's conda runtime.
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "^(AppleClang|Clang)$"
            AND NOT CMAKE_DISABLE_FIND_PACKAGE_OpenMP
            AND NOT OpenMP_omp_LIBRARY AND NOT OpenMP_libomp_LIBRARY
            AND NOT OpenMP_iomp5_LIBRARY AND NOT OpenMP_gomp_LIBRARY)
        set(_giao_conda_prefix "")
        if(GIAO_BUILD_PYTHON AND Python_EXECUTABLE)
            # The target interpreter is authoritative. CONDA_PREFIX may name
            # the base environment even when pip runs another env's Python.
            execute_process(
                COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.prefix)"
                OUTPUT_VARIABLE _giao_python_prefix OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET RESULT_VARIABLE _giao_python_status)
            if(_giao_python_status EQUAL 0 AND
                    IS_DIRECTORY "${_giao_python_prefix}/conda-meta")
                set(_giao_conda_prefix "${_giao_python_prefix}")
            endif()
        elseif(NOT GIAO_BUILD_PYTHON AND IS_DIRECTORY "$ENV{CONDA_PREFIX}/conda-meta")
            set(_giao_conda_prefix "$ENV{CONDA_PREFIX}")
        endif()

        set(_giao_omp_prefixes "${OpenMP_ROOT}" "$ENV{OpenMP_ROOT}")
        list(APPEND _giao_omp_prefixes "${_giao_conda_prefix}")
        find_program(_giao_brew brew)
        if(_giao_brew)
            execute_process(COMMAND "${_giao_brew}" --prefix libomp
                OUTPUT_VARIABLE _giao_brew_omp OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET RESULT_VARIABLE _giao_brew_status)
            if(_giao_brew_status EQUAL 0)
                list(APPEND _giao_omp_prefixes "${_giao_brew_omp}")
            endif()
        endif()
        list(APPEND _giao_omp_prefixes /opt/homebrew/opt/libomp /usr/local/opt/libomp)
        foreach(_giao_prefix IN LISTS _giao_omp_prefixes)
            if(NOT _giao_prefix)
                continue()
            endif()
            # Do not combine headers from one prefix with another runtime.
            unset(_giao_candidate_runtime CACHE)
            find_library(_giao_candidate_runtime NAMES omp iomp5
                PATHS "${_giao_prefix}/lib" "${_giao_prefix}/lib64" NO_DEFAULT_PATH)
            if(_giao_candidate_runtime AND EXISTS "${_giao_prefix}/include/omp.h")
                if(NOT OpenMP_CXX_FLAGS)
                    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp" CACHE STRING "OpenMP flags" FORCE)
                endif()
                if(NOT OpenMP_CXX_INCLUDE_DIR)
                    set(OpenMP_CXX_INCLUDE_DIR "${_giao_prefix}/include" CACHE PATH "OpenMP headers" FORCE)
                endif()
                if(NOT OpenMP_CXX_LIB_NAMES)
                    set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "OpenMP libraries" FORCE)
                endif()
                set(OpenMP_omp_LIBRARY "${_giao_candidate_runtime}" CACHE FILEPATH "OpenMP runtime" FORCE)
                message(STATUS "GIAO: using OpenMP from ${_giao_prefix}")
                break()
            elseif(_giao_prefix STREQUAL _giao_conda_prefix AND _giao_candidate_runtime)
                # Falling through to Homebrew would load a second runtime.
                set(_giao_openmp_blocked
                    "Conda OpenMP runtime exists but omp.h is missing. Install llvm-openmp in ${_giao_conda_prefix}, or select OpenMP_ROOT explicitly.")
                break()
            endif()
        endforeach()
    endif()

    if(NOT _giao_openmp_blocked)
        find_package(OpenMP QUIET COMPONENTS CXX)
    endif()
    if(OpenMP_CXX_FOUND AND NOT _giao_openmp_blocked)
        set(GIAO_OPENMP_ENABLED ON)
        message(STATUS "GIAO: OpenMP enabled (${OpenMP_CXX_LIBRARIES})")
    elseif(GIAO_ENABLE_OPENMP STREQUAL "ON")
        message(FATAL_ERROR "GIAO: OpenMP requested but not found. ${_giao_openmp_blocked} Install libomp on macOS, or set OpenMP_ROOT.")
    else()
        message(WARNING "GIAO: OpenMP unavailable; building SERIAL. ${_giao_openmp_blocked} Install libomp on macOS for parallel ERIs. GIAO_ENABLE_OPENMP=ON makes this an error.")
    endif()
else()
    message(STATUS "GIAO: SERIAL build explicitly selected (GIAO_ENABLE_OPENMP=OFF)")
endif()
