
#1.cmake verson
cmake_minimum_required(VERSION 2.8)

#2.project name
project(demo)

#3.set environment variable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set (NCNN_DIR /home/wangy/3rdParty/ncnn/build_linux/install)
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


#4. glodonfacesdk
add_library( glodonfacesdk SHARED IMPORTED )
set_target_properties(
        glodonfacesdk
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/../lib/libglodonfacesdk.so
)

message(STATUS "CMAKE_CURRENT_LIST_DIR=${CMAKE_CURRENT_LIST_DIR}/../lib")



#5. source include
include_directories(${CMAKE_CURRENT_LIST_DIR}/../include)
#6. source directory
file(GLOB SRC_CODE  ${CMAKE_CURRENT_LIST_DIR}/demo.cpp)


add_executable(demo ${SRC_CODE})

target_link_libraries(demo 
                      glodonfacesdk
                      libncnn
                      ${OpenCV_LIBS}
					  )
