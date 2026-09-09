#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "giao_integrals::core" for configuration "Release"
set_property(TARGET giao_integrals::core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(giao_integrals::core PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libgiao_integrals_core.a"
  )

list(APPEND _cmake_import_check_targets giao_integrals::core )
list(APPEND _cmake_import_check_files_for_giao_integrals::core "${_IMPORT_PREFIX}/lib/libgiao_integrals_core.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
