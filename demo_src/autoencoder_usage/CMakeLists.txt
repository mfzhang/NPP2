CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3)

project(NPP2_DEMOS CXX)

# maFIND_PACKAGE(BLAS)

add_executable(ppmautoencode ppmautoencoder.cpp)
add_dependencies(ppmautoencode npp2)

include_directories(${NPP2_SOURCE_DIR}/core ${NPP2_SOURCE_DIR}/util ${NPP2_SOURCE_DIR}/advanced ${NPP2_SOURCE_DIR}/deep ${BLAS_INCLUDE_DIRS})

target_link_libraries(ppmautoencode npp2 cblas pthread)

INSTALL(TARGETS ppmautoencode 
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
