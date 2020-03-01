cmake_minimum_required(VERSION 3.5)

project(compute_harris)

set(CMAKE_CXX_STANDARD 14)

set(OPENCV_INCLUDE_DIRS "C:\\Workstation\\libs\\opencv4_1_2\\opencv\\build\\include")
set(OPENCV_LIB_DIRS "C:\\Workstation\\libs\\opencv4_1_2\\opencv\\build\\x64\\vc15\\lib\\opencv_world412d.lib")
set(SRC src/harris.cpp src/main.cpp)

include_directories(${OPENCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} ${SRC})

add_custom_command(
	TARGET ${PROJECT_NAME}
	COMMAND ${CMAKE_COMMAND} -E copy "C:/Workstation/libs/opencv4_1_2/opencv/build/x64/vc15/bin/opencv_world412d.dll" ${CMAKE_CURRENT_BINARY_DIR}/opencv_world412d.dll
)

target_include_directories(${PROJECT_NAME} PUBLIC 
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/include/
include)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OPENCV_LIB_DIRS})
