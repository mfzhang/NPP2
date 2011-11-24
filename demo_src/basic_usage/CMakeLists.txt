CMAKE_MINIMUM_REQUIRED(VERSION 2.6.3)

project(NPP2_DEMOS CXX)

# maFIND_PACKAGE(BLAS)

add_executable(train_net train_net.cpp)
add_dependencies(train_net npp2)

include_directories(${NPP2_SOURCE_DIR}/core ${NPP2_SOURCE_DIR}/util ${BLAS_INCLUDE_DIRS})

target_link_libraries(train_net npp2 cblas pthread)

INSTALL(TARGETS train_net 
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)
