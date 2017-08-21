#pragma once

#include <vector>
#include <string>
#include <iterator>     // std::ostream_iterator
#include <algorithm>    // std::copy
#include <fstream>
#include <Eigen/Dense>

class fileHandler
{
public:
	fileHandler(){};
	~fileHandler(){};

	template<typename T>
	bool writeToFile(std::string output_file, T data){
		std::ofstream file(output_file);
		if (file.is_open()){
			file << data << '\n';
			return true;
		}
		else{
			return false;
		}

	}

	template<typename T>
	bool appendToFile(std::string file_name, T data){
		std::ofstream output_file(file_name, std::ofstream::out | std::ofstream::app);
		if (output_file.is_open()){
			output_file << data << '\n';
			return true;
		}
		else{
			return false;
		}

	}

	bool createFile(std::string);
	bool emptyFile(std::string);

};