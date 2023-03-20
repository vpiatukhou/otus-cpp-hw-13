cmake_minimum_required(VERSION 3.4.0)

set(PATCH_VERSION "1" CACHE INTERNAL "Patch version")
set(PROJECT_VERSION 0.0.${PATCH_VERSION})

project(fashio_mnist VERSION ${PROJECT_VERSION})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)

option(WITH_TESTS FALSE CACHE)

add_subdirectory(src)

if(${WITH_TESTS})
    add_subdirectory(test)
    enable_testing()
    add_test(NAME all_tests COMMAND fashio_mnist_test)
endif()