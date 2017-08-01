cmake_minimum_required(VERSION 2.8.11 FATAL_ERROR)
project( rHoughVoting)

set(WITH_RFEATURES TRUE)
include( "cmake/LibCommon.cmake")

set( INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")
set( SRC_DIR "${PROJECT_SOURCE_DIR}/src")
include_directories( "${INCLUDE_DIR}")

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

add_library( ${PROJECT_NAME} ${SRC_FILES} ${INCLUDE_FILES})
include( "cmake/LibInstall.cmake")
