cmake_minimum_required(VERSION 2.8.6)

# begin Patch swig to get concat functionality
find_package(OCTAVE REQUIRED)
include_directories (${OCTAVE_INCLUDE_DIR})
include_directories (${OCTAVE_INCLUDE_DIR}/octave)
EXECUTE_PROCESS(COMMAND ${SWIG_EXECUTABLE} -swiglib OUTPUT_VARIABLE ORIG_SWIGDIR OUTPUT_STRIP_TRAILING_WHITESPACE)

# OCTAVE front-end
add_custom_target(octave DEPENDS casadiOCT)

# Find OCTAVE
#find_package(OCTAVE REQUIRED)

include_directories(${OCTAVE_INCLUDE_DIR})
#add_definitions(${OCTAVE_EXTRA_CXXFLAGS})

# Generate SWIG wrapper
set_source_files_properties(../casadi.i  PROPERTIES  CPLUSPLUS ON)
swig_add_module(casadiOCT octave ../casadi.i)
swig_link_libraries(casadiOCT ${OCTAVE_LIBRARIES} casadi)
set_target_properties(casadiOCT PROPERTIES PREFIX "")
set_target_properties(casadiOCT PROPERTIES SUFFIX ".oct")
set_target_properties(casadiOCT PROPERTIES OUTPUT_NAME "casadi")
install(TARGETS casadiOCT DESTINATION ${CMAKE_INSTALL_PREFIX})
