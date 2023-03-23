find_package(Boost REQUIRED)

include(FetchContent)
FetchContent_Declare(
  tensorflow
  URL      https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.11.0.tar.gz
)
FetchContent_MakeAvailable(tensorflow)

add_library(fashio_mnist STATIC DataReader.cpp ImageClassifier.cpp)
add_executable(fashio_mnist_cli Main.cpp)
set_target_properties(fashio_mnist_cli PROPERTIES OUTPUT_NAME fashio_mnist) #rename the executable to "fashio_mnist"

target_include_directories(fashio_mnist PUBLIC ${tensorflow_SOURCE_DIR}/include PUBLIC ${Boost_INCLUDE_DIR})

target_link_libraries(fashio_mnist PRIVATE ${tensorflow_SOURCE_DIR}/lib/libtensorflow.so)
target_link_libraries(fashio_mnist_cli PRIVATE fashio_mnist)

#BEGIN release
install(TARGETS fashio_mnist_cli)

set(CPACK_GENERATOR DEB)

set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")

set(CPACK_PACKAGE_CONTACT example@example.com)

include(CPack)
#END
