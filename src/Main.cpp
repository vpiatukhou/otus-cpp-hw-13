#include "ImageClassifier.h"
#include "DataReader.h"

#include <iostream>
#include <stdexcept>

namespace Homework {
    const int NUMBER_OF_ARGUMENTS = 3;
    const std::size_t INPUT_DATA_PATH_INDEX = 1;
    const std::size_t MODEL_DIR_INDEX = 2;
    const std::size_t INFO_MESSAGE_EVERY_N_SAMPLES = 1000;

    const int INTERNAL_ERROR = -1;
    const int WRONG_NUMBER_OF_ARGUMENTS_ERROR = -2;
}

int main(int argc, char* argv[]) {
    using namespace Homework;

    if (argc != NUMBER_OF_ARGUMENTS) {
        std::cerr << "Wrong number of arguments.\n\tUsage: fashio_mnist input.csv path/to/tensorflow/model" << std::endl;
        return WRONG_NUMBER_OF_ARGUMENTS_ERROR;
    }

    std::size_t totalSamples = 0;
    std::size_t truePredictions = 0;

    try {
        DataReader dataReader(argv[INPUT_DATA_PATH_INDEX]);
        ImageClassifier classifier(argv[MODEL_DIR_INDEX]);

        Category category;
        Features features;
        while (dataReader.readSample(category, features)) {
            auto predictedCategory = classifier.predict(features);
            if (predictedCategory == category) {
                ++truePredictions;
            }

            ++totalSamples;

            if (totalSamples % INFO_MESSAGE_EVERY_N_SAMPLES == 0) {
                std::cout << "Processed samples: " << totalSamples << std::endl;
            }
        }
    } catch (InvalidDataException& e) {
        std::cerr << "Error reading a sample #" << (totalSamples + 1) << ": " << e.what() << std::endl;
        return INTERNAL_ERROR;
    } catch (InvalidModelException& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return INTERNAL_ERROR;
    } catch (std::exception& e) {
        std::cerr << "Internal error: " << e.what() << std::endl;
        return INTERNAL_ERROR;
    }

    double accuracy = static_cast<double>(truePredictions) / totalSamples;
    std::cout << truePredictions << " out of " << totalSamples << " predictions are correct.\nAccuracy: " << accuracy << std::endl;

    return 0;
}
