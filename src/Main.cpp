#include "ImageClassifier.h"
#include "DataReader.h"

#include <iostream>
#include <stdexcept>

namespace Homework {
    const int NUMBER_OF_ARGUMENTS = 3;
    const std::size_t INPUT_DATA_PATH_INDEX = 1;
    const std::size_t MODEL_DIR_INDEX = 2;

    const std::size_t CATEGORY_INDEX = 0;
    const std::size_t NUMBER_OF_FEATURES = 28 * 28;

    const float MAX_FEATURE_VALUE = 255.0f;
}

int main(int argc, char* argv[]) {
    using namespace Homework;

    if (argc != NUMBER_OF_ARGUMENTS) {
        std::cerr << "Wrong number of arguments.\n\tUsage: fashio_mnist input.csv path/to/tensorflow/model" << std::endl;
        return -2;
    }

    Category category;

    std::size_t totalSamples = 0;
    std::size_t truePredictions = 0;

    try {
        DataReader dataReader(argv[INPUT_DATA_PATH_INDEX]);   //  /app/data/test.csv      //TODO to remove
        ImageClassifier classifier(argv[MODEL_DIR_INDEX]);         //  /app/data/saved_model/  //TODO to remove
        std::size_t lineNumber = 0;
        while (dataReader.hasNext()) {
            Category category;
            Features features;
            try {
                dataReader.readSample(category, features);
            } catch (std::invalid_argument& e) {
                std::cout << "Error on reading a sample #" << lineNumber << ": " << e.what() << std::endl;
            }

            if (features.empty()) { //TODO refactor
                break;
            }

            auto predictedCategory = classifier.predict(features);

            ++totalSamples;
            if (category == predictedCategory) {
                ++truePredictions;
            }

            if (totalSamples % 1000 == 0) {
                std::cout << "Processed samples: " << totalSamples << std::endl;
            }

            ++lineNumber;
        }
    } catch (std::exception& e) {
        std::cerr << "Internal error: " << e.what() << std::endl;
        return -1;
    }

    double accuracy = static_cast<double>(truePredictions) / totalSamples;
    std::cout << truePredictions << " out of " << totalSamples << " predictions are correct.\nAccuracy: " << accuracy << std::endl;

    return 0;
}
