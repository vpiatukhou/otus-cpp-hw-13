#pragma once

#include "Types.h"

#include <fstream>
#include <string>

namespace Homework {

    class DataReader {
    public:
        DataReader(const std::string& filepath_);

        bool hasNext();
        void readSample(Category& category, Features& features);

    private:
        std::ifstream file;
    };

}
