#include <grasp_learning/fileHandler.h>

bool fileHandler::writeToFile(std::string output_file, std::vector<double> data){

	return true;
}

bool fileHandler::appendToFile(std::string file_name, std::vector<double> data){
	 std::ofstream output_file(file_name);

	std::ostream_iterator<double> output_iterator(output_file, "\n");
	std::copy(data.begin(), data.end(), output_iterator);
	return true;
}

bool fileHandler::createFile(std::string output_file){
	std::ofstream ofs(output_file);
	return true;
}

bool fileHandler::emptyFile(std::string output_file){
	std::ofstream ofs;
	ofs.open(output_file, std::ofstream::out | std::ofstream::trunc);
	ofs.close();
	return true;
}
