cmake_minimum_required(VERSION 3.5)

project(compute_harris)

set(CMAKE_CXX_STANDARD 14)

set(OPENCV_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/include")
set(OPENCV_LIB_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/lib/opencv_world412d.lib")
set(SRC src/harris.cpp src/main.cpp)

include_directories(${OPENCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${SRC})

add_custom_command(
	TARGET ${PROJECT_NAME}
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/bin/opencv_world412d.dll" ${CMAKE_CURRENT_BINARY_DIR}/opencv_world412d.dll
)

target_include_directories(${PROJECT_NAME} PUBLIC 
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/include/
include)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OPENCV_LIB_DIRS})
