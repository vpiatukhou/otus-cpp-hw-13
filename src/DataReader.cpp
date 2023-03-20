#include "DataReader.h"

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

namespace Homework {

    namespace {
        const std::string DELIMITER = ",";
        const float MAX_FEATURE_VALUE = 255.0f;
    }

    DataReader::DataReader(const std::string& filepath_) : file(filepath_) {
    }

    bool DataReader::hasNext() {
        return !file.eof();
    }

    void DataReader::readSample(Category& category, Features& features) {
        std::string line;
        std::getline(file, line);

        if (line.empty()) {
            return;
        }
        
        std::vector<std::string> values;
        boost::algorithm::split(values, line, boost::is_any_of(DELIMITER));

        features.reserve(values.size() - 1);

        try {
            category = std::stoi(values[0]);
        } catch (...) {
            throw std::invalid_argument("Could not convert category " + values[0] + " to integer.");
        }

        for (std::size_t i = 1; i < values.size(); ++i) {
            int feature;
            try {
                feature = std::stoi(values[i]);
            } catch (...) {
                throw std::invalid_argument("Could not convert a feature " + values[i] + " to integer.");
            }
            auto normalizedFeature = feature / MAX_FEATURE_VALUE;
            features.push_back(normalizedFeature);
        }
    }

}
