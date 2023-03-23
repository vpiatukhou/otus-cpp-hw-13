#include "DataReader.h"

#include "gmock/gmock.h"

#include <fstream>
#include <string>

namespace {
    using namespace Homework;
    using ::testing::ElementsAre;

    const std::string TEMP_FILE = "temp.txt";

    /**
     * Creates and deletes test data.
     */
    class DataReaderTest : public ::testing::Test {
    protected:
        ~DataReaderTest() {
            std::remove(TEMP_FILE.c_str());
        }

        void setUp(const std::string& testData) {
            std::ofstream testFile(TEMP_FILE);
            testFile << testData;
        }
    };

    void verifyFeatures(const Features& expected, const Features& actual) {
        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_FLOAT_EQ(expected[i], actual[i]);
        }
    }
}

TEST_F(DataReaderTest, readSample_positive) {
    //given
    setUp("7,0,50,255,128\n0,128, 2 ,9,255\n5,1,2,3,4");

    DataReader reader(TEMP_FILE);

    Category category1, category2, category3;
    Features features1, features2, features3;

    //when
    auto result1 = reader.readSample(category1, features1);
    auto result2 = reader.readSample(category2, features2);
    auto result3 = reader.readSample(category3, features3);

    //then
    ASSERT_TRUE(result1);
    ASSERT_EQ(7, category1);
    Features expected1 = {0, 50.0f / 255.0f, 1.0f, 128.0f / 255.0f};
    verifyFeatures(expected1, features1);

    ASSERT_TRUE(result2);
    ASSERT_EQ(0, category2);
    Features expected2 = {128.0f / 255.0f, 2.0f / 255.0f, 9.0f / 255.0f, 1.0f};
    verifyFeatures(expected2, features2);

    ASSERT_TRUE(result3);
    ASSERT_EQ(5, category3);
    Features expected3 = {1.0f / 255.0f, 2.0f / 255.0f, 3.0f / 255.0f, 4.0f / 255.0f};
    verifyFeatures(expected3, features3);
}

TEST_F(DataReaderTest, readSample_emptyLine) {
    //given
    setUp("7,3,12\n1,0,255\n");

    DataReader reader(TEMP_FILE);

    Category category1, category2, category3;
    Features features1, features2, features3;

    //when
    auto result1 = reader.readSample(category1, features1);
    auto result2 = reader.readSample(category2, features2);
    auto result3 = reader.readSample(category3, features3);

    //then
    ASSERT_TRUE(result1);
    ASSERT_TRUE(result2);
    ASSERT_FALSE(result3);

    ASSERT_EQ(1, category2);
    Features expected = {0, 1.0f};
    verifyFeatures(expected, features2);
}

TEST_F(DataReaderTest, readSample_categoryGreaterThanLimit) {
    setUp("10,0,0");

    Category category;
    Features features;
    DataReader reader(TEMP_FILE);

    ASSERT_THROW(reader.readSample(category, features), InvalidDataException);
}

TEST_F(DataReaderTest, readSample_categoryLessThanLimit) {
    setUp("-1,0,0");

    Category category;
    Features features;
    DataReader reader(TEMP_FILE);

    ASSERT_THROW(reader.readSample(category, features), InvalidDataException);
}

TEST_F(DataReaderTest, readSample_featureGreaterThanLimit) {
    setUp("0,256,0");

    Category category;
    Features features;
    DataReader reader(TEMP_FILE);

    ASSERT_THROW(reader.readSample(category, features), InvalidDataException);
}

TEST_F(DataReaderTest, readSample_featureLessThanLimit) {
    setUp("0,-1,0");

    Category category;
    Features features;
    DataReader reader(TEMP_FILE);

    ASSERT_THROW(reader.readSample(category, features), InvalidDataException);
}
