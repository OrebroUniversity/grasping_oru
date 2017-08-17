#include <grasp_learning/fileHandler.h>

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
