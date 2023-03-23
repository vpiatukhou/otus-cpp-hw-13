#pragma once

#include "Types.h"

#include <fstream>
#include <stdexcept>
#include <string>

namespace Homework {

    /**
     * Thorwn if the CSV file contains invalid data.
     */
    class InvalidDataException : public std::runtime_error {
    public:
        InvalidDataException(const std::string& message) : std::runtime_error(message) { }
    };

    /**
     * Reads test data from CSV file.
     * 
     * The CSV files must meed the requirements:
     * - no headers;
     * - a delimiter is a comma;
     * - the first column contains categories. Other columns contains features;
     * - the categories are in the range of [0..9];
     * - the features are in the range of [0..255].
     * 
     * If the CSV doesn't meet the requirements, the behaviour is underfined.
     */
    class DataReader {
    public:
        DataReader(const std::string& filepath_);

        /**
         * Reads a category and a list of features from the current line of the CSV file.
         * 
         * @param category - an output parameter. A category of the sample. It is always in range of [0..9].
         *                   The value of the parameter doesn't changed if the method returns FALSE.
         * @param features - features of the sample. All values are normalized (they are in the range of [0..1]).
         *                   The value of the parameter doesn't changed if the method returns FALSE.
         * @return TRUE if the sample has been read. FALSE if EOF or an empty line has been reached.
         * @throws InvalidDataException if a category or a feature is out of range or not a number
         */
        bool readSample(Category& category, Features& features);

    private:
        std::ifstream file;
    };

}
