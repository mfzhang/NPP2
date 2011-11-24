CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3)

project(NPP2 CXX C)

INCLUDE( configuration.cmake.in )

# find optional packages
MESSAGE( STATUS )
MESSAGE( STATUS "Searching for additional packages ---------------------------------------------" )
FIND_PACKAGE(Doxygen)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Doxygen DEFAULT_MSG DOXYGEN_EXECUTABLE)

MESSAGE (STATUS ${CMAKE_OSX_ARCHITECTURES})

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
SET(CMAKE_INSTALL_PREFIX
"${NPP2_SOURCE_DIR}" CACHE PATH "NPP2 install prefix" FORCE
)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

OPTION( DEMOS "Set to OFF to prevent compilation of demos." OFF )

MESSAGE( STATUS )
MESSAGE( STATUS "--N++2 package options ----------------------------------------------------" )
MESSAGE( STATUS "CMAKE_INSTALL_PREFIX   = ${CMAKE_INSTALL_PREFIX}" )
MESSAGE( STATUS "DEMOS                  = ${DEMOS}" )
MESSAGE( STATUS "CMAKE_BUILD_TYPE       = ${CMAKE_BUILD_TYPE}")
MESSAGE( STATUS "Change a value with: cmake -D<VAR>=<VALUE>" )
MESSAGE( STATUS "-------------------------------------------------------------------------------" )
MESSAGE( STATUS )

SET( DEMOS "${DEMOS}" CACHE BOOL "Set to OFF to prevent compilation of demos." FORCE )

add_subdirectory(src)

IF ( DEMOS )
add_subdirectory(demo_src)
ENDIF()

# doc target
IF(DOXYGEN_FOUND)
  ADD_CUSTOM_TARGET(doc COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE} 
	                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
ENDIF()
