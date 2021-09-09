
#1.cmake verson
cmake_minimum_required(VERSION 2.8)

set (NCNN_DIR /home/wangy/3rdParty/ncnn/build_linux/install)

#2.project name
project(facesdk)


#3.set environment variable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


#4. dependence 

## ncnn include
include_directories(${NCNN_DIR}/include/ncnn)

## ncnn lib
add_library( libncnn STATIC IMPORTED )
set_target_properties(
        libncnn
        PROPERTIES IMPORTED_LOCATION
        ${NCNN_DIR}/lib/libncnn.a
)
## opencv 
find_package(OpenCV REQUIRED)


## OpenMP
#find_package(OpenMP REQUIRED)
#if(OPENMP_FOUND)
#    message(STATUS "OpenMP found")
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    message(STATUS "CMAKE_CXX_FLAGS=${OpenMP_CXX_FLAGS}")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

#5. source include 
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
message(STATUS "CMAKE_CURRENT_LIST_DIR=${CMAKE_CURRENT_LIST_DIR}")

#6.source directory
file(GLOB SRC_CODE  ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)


#7. set so output path
SET(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/)
message(STATUS "CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")

#8. generate so
add_library(facesdk SHARED ${SRC_CODE})

#9. add dependence
target_link_libraries(facesdk 
                      libncnn
                      ${OpenCV_LIBS}
					  )
					  


