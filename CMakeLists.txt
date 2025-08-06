cmake_minimum_required(VERSION 3.10.0)
project(detection VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(OpenCV_DIR "/Users/Opencv/opencv_build")
find_package(OpenCV REQUIRED)

include_directories("/Users/m_openvino_toolkit_macos_11_0_2023.3.0.13775.ceeafaf64f3_arm64/runtime/include")
set(OpenVINO_DIR "/Users/m_openvino_toolkit_macos_11_0_2023.3.0.13775.ceeafaf64f3_arm64/runtime/cmake")
find_package(OpenVINO REQUIRED)
link_directories("/Users/m_openvino_toolkit_macos_11_0_2023.3.0.13775.ceeafaf64f3_arm64/runtime/lib/arm64/Release") 


# find_package(OpenCV REQUIRED COMPONENTS core dnn)
# find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
# find_package(OpenVINO REQUIRED COMPONENTS runtime)

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS} 
    ${OpenVINO_INCLUDE_DIRS}
)

include_directories(/Users/linzenggeng/Desktop/detection/include)
include_directories(/Users/linzenggeng/Desktop/detection/TimeCounter)
include_directories(/Users/linzenggeng/Desktop/detection/TrackerHeader)
include_directories(/Users/linzenggeng/Desktop/detection/Eigen)

file(GLOB_RECURSE SOURCES "TrackerSource/*.cpp")
add_executable(${PROJECT_NAME} main.cpp ${SOURCES})
# message(${SOURCES})


target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${OpenCV_INCLUDE_DIRS}
    ${OpenVINO_LIBRARIES}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenVINO_LIBRARIES}  
    ${OpenCV_LIBS}               # OpenCV 库
    openvino::runtime
)
