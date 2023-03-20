#pragma once

#include <string>

#include <tensorflow/c/c_api.h>

#include <vector>
#include <memory>

namespace Homework {

    using Category = std::uint8_t;
    using Features = std::vector<float>;

    class ImageClassifier {
    public:
        ImageClassifier(const std::string& modelDir_);
        ImageClassifier(const ImageClassifier&) = delete;
        ImageClassifier(ImageClassifier&&) = delete;

        ImageClassifier& operator=(const ImageClassifier&) = delete;
        ImageClassifier& operator=(ImageClassifier&&) = delete;

        Category predict(Features& input);

    private:
        std::vector<TF_Output> inputs;
        std::vector<TF_Output> outputs;

        static void deleteSession(TF_Session* tfSession);

        using TfGraph = std::unique_ptr<TF_Graph, decltype(&TF_DeleteGraph)>;
        using TfStatus = std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)>;
        using TfSessionOptions = std::unique_ptr<TF_SessionOptions, decltype(&TF_DeleteSessionOptions)>;
        using TfSession = std::unique_ptr<TF_Session, decltype(&deleteSession)>;
        using TfTensor = std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)>;

        TfGraph graph{TF_NewGraph(), TF_DeleteGraph};
        TfStatus status{TF_NewStatus(), TF_DeleteStatus};
        TfSessionOptions sessionOptions{TF_NewSessionOptions(), TF_DeleteSessionOptions};
        TfSession session{nullptr, deleteSession};
    };

}