cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(ncnn-mat-test LANGUAGES CXX)

set(ncnn_DIR /home/qingren/Project/GitHub/ncnn/build-20220711/install/lib/cmake/ncnn)
find_package(ncnn REQUIRED)


macro(add_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} ncnn)
endmacro()

add_test(ncnn-vector-test)
add_test(ncnn-matrix-test)
add_test(ncnn-cube-test)

