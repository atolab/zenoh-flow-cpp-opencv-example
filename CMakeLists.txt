#
# Copyright (c) 2017, 2021 ADLINK Technology Inc.
#
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# http://www.eclipse.org/legal/epl-2.0, or the Apache License, Version 2.0
# which is available at https://www.apache.org/licenses/LICENSE-2.0.
#
# SPDX-License-Identifier: EPL-2.0 OR Apache-2.0
#
# Contributors:
#   ADLINK zenoh team, <zenoh@adlink-labs.tech>
#

cmake_minimum_required(VERSION 3.10)

project(ZenohFlowCxxNode VERSION 0.1.0)

include(FetchContent)

FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json
  GIT_TAG v3.7.3)

FetchContent_GetProperties(json)
  if(NOT json_POPULATED)
    FetchContent_Populate(json)
    add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()

# TODO: Detect that we are running on a Apple M1 and set this variable to this value.
# https://gitlab.kitware.com/cmake/cmake/-/issues/20989
# set(CMAKE_OSX_ARCHITECTURES arm64)
# message("-- CMAKE_OSX_ARCHITECTURES = ${CMAKE_OSX_ARCHITECTURES}")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  set(CARGO_BUILD_TYPE --release)
  set(CARGO_BUILD_TYPE_DIR release)
endif()

message("-- CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

string( TOLOWER "${CMAKE_BUILD_TYPE}" cm_build_type )
if(cm_build_type STREQUAL "debug" )
  set(CARGO_BUILD_TYPE_DIR debug)
endif()

message("-- CARGO_BUILD_TYPE = ${CARGO_BUILD_TYPE_DIR}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# TODO: Windows compatibility.
find_program(CXXBRIDGE cxxbridge REQUIRED PATHS $ENV{HOME}/.cargo/bin)
find_package(OpenCV REQUIRED )
message(STATUS "Using cxxbridge: ${CXXBRIDGE}")

set(CMAKE_SCRIPTS_DIR ${CMAKE_SOURCE_DIR}/cmake)

#
# Setup: We need to know what we are generating.
#
option(SOURCE    "Set the Node to SOURCE")
option(OPERATOR  "Set the Node to OPERATOR")
option(SINK      "Set the Node to SINK")


if (NOT ${SOURCE} AND NOT ${OPERATOR} AND NOT ${SINK})
  message(FATAL_ERROR "The type of Node to generate must be specified (i.e., -DOPERATOR=ON, -DSINK=ON, -DSOURCE=ON).")
elseif ((${SOURCE} AND (${OPERATOR} OR ${SINK})) OR (${OPERATOR} AND (${SOURCE} OR ${SINK})))
  message(FATAL_ERROR "Only one type of Node can be generated.")
endif()

if (${SOURCE})
  message(FATAL_ERROR "Only Operator node can be generated from this code.")
elseif (${SINK})
  message(FATAL_ERROR "Only Operator node can be generated from this code.")
elseif (${OPERATOR})
  set(node operator)
endif()

set(LIB_NAME "cxx_${node}" CACHE STRING "The name of the generated library.")

#
# First step: generating the bridge files.
#
# The generated lib.rs file will be parsed by `cxxbridge` to generate the
# bindings between Zenoh Flow and the C++ nodes.
#
set(node_dir  ${CMAKE_SOURCE_DIR}/vendor/${node})

set(cxxbridge_in         ${node_dir}/src/lib.rs)
set(cxxbridge_source_out ${CMAKE_SOURCE_DIR}/src/wrapper.cpp)
set(cxxbridge_header_out ${CMAKE_SOURCE_DIR}/include/wrapper.hpp)

add_custom_target(CxxBridge ALL
  DEPENDS ${cxxbridge_header_out} ${cxxbridge_source_out}
  )

add_custom_command(
  OUTPUT ${cxxbridge_header_out} ${cxxbridge_source_out}
  COMMAND ${CXXBRIDGE} ${cxxbridge_in} --output ${cxxbridge_source_out}
  COMMAND ${CXXBRIDGE} ${cxxbridge_in} --header --output ${cxxbridge_header_out}
  COMMAND ${CMAKE_COMMAND} -DHEADER=${cxxbridge_header_out} -P ${CMAKE_SCRIPTS_DIR}/Patcher.cmake
  COMMENT "Generating CXX bridge:"
  )

#
# Second step: generating the static library containing the Rust "glue".
#
set(rust_lib ${node_dir}/target/${CARGO_BUILD_TYPE_DIR}/lib${node}.a)

add_custom_target(RustLib ALL DEPENDS ${rust_lib})
add_custom_command(
  OUTPUT ${rust_lib}
  # COMMAND cargo update
  COMMAND cargo build ${CARGO_BUILD_TYPE}
  WORKING_DIRECTORY ${node_dir}
  DEPENDS ${cxxbridge_in}
  COMMENT "Generating Rust library:"
  )

add_library(CxxWrapper STATIC IMPORTED GLOBAL)
add_dependencies(CxxWrapper RustLib)
set_target_properties(CxxWrapper
  PROPERTIES
  IMPORTED_LOCATION ${rust_lib}
  )

#
# Third step: generating the final node shared library.
#
set(node_src src/${node}.cpp)

add_library(${LIB_NAME} SHARED
  ${node_src}
  ${cxxbridge_source_out})
add_dependencies(${LIB_NAME} CxxBridge)
target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(${LIB_NAME} PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(${LIB_NAME}  PUBLIC ${OpenCV_LIBS} )
target_link_libraries(${LIB_NAME} PUBLIC CxxWrapper)
