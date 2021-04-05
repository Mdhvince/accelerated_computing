cmake_minimum_required(VERSION 3.5)

project(SudoAir VERSION 1.0.0)

add_definitions(-std=c++17)
set(CXX_FLAGS "-Wall")

#OpenCV
find_package(OpenCV REQUIRED)

set(EXECUTABLE app)
add_executable(${EXECUTABLE} camera.cpp)

target_link_libraries(${EXECUTABLE} PRIVATE ${OpenCV_LIBS})
