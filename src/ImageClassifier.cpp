#include "ImageClassifier.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace Homework {

    namespace {
        const char* TAGS = "serve";
        const int TAGS_LENGTH = 1;

        const std::size_t IMAGE_SIDE_SIZE = 28;
        const std::size_t IMAGE_SIZE = IMAGE_SIDE_SIZE * IMAGE_SIDE_SIZE;
        const std::uint8_t NUMBER_OF_CATEGORIES = 10;
    }

    static void emptyDeallocator(void* data, size_t length, void* arg) {
    }

    static Category probabilitiesToCategory(float* probabilies) {
        float maxValue = 0;
        Category result = 0;
        for (Category i = 0; i < NUMBER_OF_CATEGORIES; ++i) {
            if (maxValue < probabilies[i]) {
                maxValue = probabilies[i];
                result = i;
            }
        }
        return result;
    }

    ImageClassifier::ImageClassifier(const std::string& modelDir_) {
        session.reset({TF_LoadSessionFromSavedModel(sessionOptions.get(), nullptr, modelDir_.c_str(), 
            &TAGS, TAGS_LENGTH, graph.get(), nullptr, status.get())});

        //inputs
        auto inputOperation = TF_GraphOperationByName(graph.get(), "serving_default_input");
        if (inputOperation == nullptr) {
            throw std::runtime_error("An input operation was not found.");
        }
        inputs = {{inputOperation, 0}};        

        //outputs
        auto outputOperation = TF_GraphOperationByName(graph.get(), "StatefulPartitionedCall");
        if (outputOperation == nullptr) {
            throw std::runtime_error("An output operation was not found");
        }
        outputs = {{outputOperation, 0}};
    }

    Category ImageClassifier::predict(Features& input_) {
        int64_t dimensions[] = {1, IMAGE_SIDE_SIZE, IMAGE_SIDE_SIZE, 1};
        const int inputSizeInBytes = IMAGE_SIZE * sizeof(float);
        TfTensor inputTensor{TF_NewTensor(TF_FLOAT, dimensions, 4, 
                                          reinterpret_cast<void*>(input_.data()), inputSizeInBytes, &emptyDeallocator, 0), 
                                          TF_DeleteTensor};
        std::vector<TF_Tensor*> inputValues = {inputTensor.get()};

        std::vector<TF_Tensor*> outputValues;
        outputValues.resize(1);

        TF_SessionRun(session.get(), nullptr, 
            &inputs[0], &inputValues[0], inputs.size(), 
            &outputs[0], &outputValues[0], outputs.size(), 
            nullptr, 0, nullptr, status.get());
        if (TF_GetCode(status.get()) != TF_OK) {
            std::string msg = "Unable to run session from graph: ";
            msg += TF_Message(status.get());
            throw std::runtime_error(msg);
        }

        float* probabilies = static_cast<float*>(TF_TensorData(outputValues[0]));
        return probabilitiesToCategory(probabilies);
    }

    void ImageClassifier::deleteSession(TF_Session* tfSession) {
        TfStatus status{TF_NewStatus(), TF_DeleteStatus};
        TF_DeleteSession(tfSession, status.get());
        if (TF_GetCode(status.get()) != TF_OK) {
            std::cerr << "Error deleting the tensorflow session. Status: " << TF_Message(status.get()) << std::endl;
        }
    }

}