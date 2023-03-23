#include "DataReader.h"

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

namespace Homework {

    namespace {
        const std::string DELIMITER = ",";
        const float MAX_FEATURE_VALUE = 255.0f;
        const int MAX_CATEGORY_VALUE = 9;
    }

    DataReader::DataReader(const std::string& filepath_) : file(filepath_) {
    }

    bool DataReader::readSample(Category& category, Features& features) {
        std::string line;

        if (!std::getline(file, line) || line.empty()) {
            return false;
        }
        
        std::vector<std::string> values;
        boost::algorithm::split(values, line, boost::is_any_of(DELIMITER));

        features.clear();
        features.reserve(values.size() - 1);

        try {
            category = std::stoi(values[0]);
        } catch (...) {
            throw InvalidDataException("Could not convert category " + values[0] + " to integer.");
        }

        if (category < 0 || category > MAX_CATEGORY_VALUE) {
            throw InvalidDataException("Category " + std::to_string(category) + " must be in the range [0..9].");
        }

        for (std::size_t i = 1; i < values.size(); ++i) {
            int feature;
            try {
                feature = std::stoi(values[i]);
            } catch (...) {
                throw InvalidDataException("Could not convert a feature " + values[i] + " to integer.");
            }

            if (feature < 0 || feature > MAX_FEATURE_VALUE) {
                throw InvalidDataException("A feature " + std::to_string(category) + " must be in the range [0..255].");
            }

            auto normalizedFeature = feature / MAX_FEATURE_VALUE;
            features.push_back(normalizedFeature);
        }
        return true;
    }

}
