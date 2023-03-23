set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

add_executable(fashio_mnist_test DataReaderTest.cpp)

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)

FetchContent_MakeAvailable(googletest)

target_include_directories(fashio_mnist_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(fashio_mnist_test fashio_mnist GTest::gtest GTest::gtest_main GTest::gmock_main)

include(GoogleTest)
gtest_discover_tests(fashio_mnist_test)
