#pragma once

#include <vector>
#include <string>
#include <iterator>     // std::ostream_iterator
#include <algorithm>    // std::copy
#include <fstream>
#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <iostream>
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

	bool createFolder(std::string dir_path){
		boost::filesystem::path dir(dir_path);
		if(!boost::filesystem::exists(dir_path)){
			if(!boost::filesystem::create_directory(dir)) {
				return false;
			}
		}
		return true;
	}


	bool createFolders(std::vector<std::string> dir_paths){

		for(auto& dir_path:dir_paths){
			if(!boost::filesystem::exists(dir_path)){
				boost::filesystem::path dir(dir_path);
				if(!boost::filesystem::create_directory(dir)) {
					return false;
				}
			}
		}
		return true;
	}

	bool createFile(std::string);
	bool emptyFile(std::string);

};