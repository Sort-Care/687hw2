cmake_minimum_required(VERSION 2.8.9)
project(687hw2)
set(CMAKE_CXX_STANDARD 11)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "\\.#.*")

add_executable(hw2 ${SOURCES})

target_link_libraries (hw2 Eigen3::Eigen)
