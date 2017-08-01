# This module defines:
#  rHoughVoting_ROOT_DIR - the root directory where the library is installed.
#  rHoughVoting_INCLUDE_DIR - the include directory.
#  rHoughVoting_LIBRARY_DIR - the library directory.
#  rHoughVoting_LIBRARY - library to link to.

get_filename_component( _prefix "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component( rHoughVoting_ROOT_DIR "${_prefix}" PATH)

set( rHoughVoting_INCLUDE_DIR "${rHoughVoting_ROOT_DIR}/include")
set( rHoughVoting_LIBRARY_DIR "${rHoughVoting_ROOT_DIR}/lib")

include( "${CMAKE_CURRENT_LIST_DIR}/Macros.cmake")
get_library_suffix( _lib_suffix)

set( rHoughVoting_LIBRARY rHoughVoting${_lib_suffix})
