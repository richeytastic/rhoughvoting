cmake_minimum_required(VERSION 2.8.11 FATAL_ERROR)
project( rHoughVoting)

include( "../LibCMakeCommon.cmake")
set( BUILD_SHARED_LIBS TRUE)
set( SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set( INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
include_directories( "${INCLUDE_DIR}")
# Include the build directory because generated header files sit here.
include_directories( "${PROJECT_BINARY_DIR}")

# Boost
if(WIN32)
    set( BOOST_ROOT "D:/boost_1_56_0")
    set( BOOST_LIBRARYDIR "${BOOST_ROOT}/lib64-msvc-11.0")
endif(WIN32)
find_package( Boost 1.5 REQUIRED COMPONENTS filesystem system thread random)
include_directories( ${Boost_INCLUDE_DIRS})
link_directories( ${Boost_LIBRARY_DIRS})

# OpenCV
if(WIN32)
    set( OpenCV_DIR "${LOCALLIBS}/opencv")
endif(WIN32)
find_package( OpenCV 2.4 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set( RFeatures_DIR "${LOCALLIBS}/rFeatures")
find_package( RFeatures REQUIRED)
include_directories( ${RFeatures_INCLUDE_DIR})

set( LibLAS_DIR "${LOCALLIBS}/libLAS-1.7.0")
find_package( LibLAS REQUIRED)

set( RLIB_DIR "${LOCALLIBS}/rlib")
find_package( RLIB REQUIRED)
include_directories( ${RLIB_INCLUDE_DIR})


set( INCLUDE_FILES
    "${INCLUDE_DIR}/Forest.h"
    "${INCLUDE_DIR}/ForestScanner.h"
    "${INCLUDE_DIR}/NonMaxSuppressor.h"
    "${INCLUDE_DIR}/RandomForest.h"
    "${INCLUDE_DIR}/RandomForest_MT.h"
    "${INCLUDE_DIR}/RandomTree.h"
    "${INCLUDE_DIR}/RHoughVoting.h"
    #"${INCLUDE_DIR}/SupportSuppressor.h"
    "${INCLUDE_DIR}/TestParamGenerator.h"
    "${INCLUDE_DIR}/Tree.h"
    #"${INCLUDE_DIR}/TreeScanner.h"
	)

set( SRC_FILES
    ${SRC_DIR}/Forest
    ${SRC_DIR}/ForestScanner
    ${SRC_DIR}/NonMaxSuppressor
    ${SRC_DIR}/RandomForest
    ${SRC_DIR}/RandomForest_MT
    ${SRC_DIR}/RandomTree
    #${SRC_DIR}/SupportSuppressor
    ${SRC_DIR}/TestParamGenerator
    ${SRC_DIR}/Tree
    #${SRC_DIR}/TreeScanner
	)


# Require generate_export_header for .lib import libraries associated with Windows DLLs (if BUILD_SHARED_LIBS is true)
include(GenerateExportHeader)
add_library( ${PROJECT_NAME} ${SRC_FILES} ${INCLUDE_FILES})
generate_export_header( ${PROJECT_NAME}
    BASE_NAME ${PROJECT_NAME}
    EXPORT_MACRO_NAME ${PROJECT_NAME}_EXPORT
    EXPORT_FILE_NAME ${PROJECT_NAME}_Export.h
    STATIC_DEFINE ${PROJECT_NAME}_BUILT_AS_STATIC)

target_link_libraries( ${PROJECT_NAME} ${Boost_LIBRARIES})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries( ${PROJECT_NAME} ${RFeatures_LIBRARY})
target_link_libraries( ${PROJECT_NAME} ${LibLAS_LIBRARY})
target_link_libraries( ${PROJECT_NAME} ${RLIB_LIBRARY})

set( CMAKE_INSTALL_PREFIX "${LOCALLIBS}/${PROJECT_NAME}")
install( TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "bin"
                                 LIBRARY DESTINATION "lib"
                                 ARCHIVE DESTINATION "lib/static")

install( FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}_Export.h" DESTINATION "include")
install( FILES ${INCLUDE_FILES} DESTINATION "include")
# Copy across this library's Find cmake module to ${LOCALLIBS}/cmake
install( FILES "${PROJECT_SOURCE_DIR}/../CMakeModules/FindRHoughVoting.cmake" DESTINATION "${LOCALLIBS}/cmake")



