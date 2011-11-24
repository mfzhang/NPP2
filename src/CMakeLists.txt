CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3)

project(NPP2 CXX)

# if no option is given, standard is release
IF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE Release)
ENDIF(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_FLAGS_RELEASE " -Wall -O2")
set(CMAKE_CXX_FLAGS_DEBUG " -Wall -pg")

set(NPP2_LIB "")

OPTION( INSTALL_LIBS        "Set to ON for explicit installation of libraries." OFF )


### Add neural core sources
include(${NPP2_SOURCE_DIR}/core/Sources.cmake)

### Add util sources 
include(${NPP2_SOURCE_DIR}/util/Sources.cmake)

### Add deep sources 
include(${NPP2_SOURCE_DIR}/deep/Sources.cmake)

### Add advanced sources 
include(${NPP2_SOURCE_DIR}/advanced/Sources.cmake)


# display status message for important variables
MESSAGE( STATUS )
MESSAGE( STATUS "--CLSquare options ----------------------------------------------------" )
MESSAGE( STATUS "INSTALL_LIBS          = ${INSTALL_LIBS}" )
MESSAGE( STATUS "Change a value with: cmake -D<VAR>=<VALUE>" )
MESSAGE( STATUS "-------------------------------------------------------------------------------" )
MESSAGE( STATUS )

# force some variables that could be defined in the command line to be written to cache
SET( INSTALL_LIBS "${INSTALL_LIBS}" CACHE BOOL "Set to ON to install libraries." FORCE )

# define subgroups for XCode and other IDEs
source_group( Core FILES ${core_headers} ${core_srcs} )
source_group( Util FILES ${util_headers} ${util_srcs} )
source_group( Deep FILES ${deep_headers} ${deep_srcs} )
source_group( Advanced FILES ${advanced_headers} ${advanced_srcs} )

# preprocess the dependencies
list(REMOVE_DUPLICATES NPP2_LIB)

# display status message for dependencies
MESSAGE( STATUS )
MESSAGE( STATUS "-------------------------------------------------------------------------------" )
MESSAGE( STATUS "linking with standard libs NPP2_LIB = ${NPP2_LIB}" )
MESSAGE( STATUS "-------------------------------------------------------------------------------" )
MESSAGE( STATUS )

ADD_LIBRARY(npp2 STATIC ${core_srcs} ${util_srcs} ${advanced_srcs} ${deep_srcs}) 

INSTALL(FILES ${core_headers} ${util_headers} ${advanced_headers} ${deep_headers} DESTINATION include)

INSTALL(TARGETS npp2 
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)


