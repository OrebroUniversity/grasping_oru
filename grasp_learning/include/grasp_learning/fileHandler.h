#pragma once

#include <vector>
#include <string>
#include <iterator>     // std::ostream_iterator
#include <algorithm>    // std::copy
#include <fstream>

class fileHandler
{
public:
	fileHandler(){};
	~fileHandler(){};

	bool writeToFile(std::string, std::vector<double> );
	bool createFile(std::string);
	bool emptyFile(std::string);
	bool appendToFile(std::string, std::vector<double>);
};