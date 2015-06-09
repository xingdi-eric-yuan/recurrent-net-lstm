#pragma once
#include <fstream>
#include "general_settings.h"

using namespace std;
using namespace cv;

void readWordvec(std::string, unordered_map<string, Mat>&);

void
readDataset(std::string, std::string, 
            std::string, std::string, 
            std::vector<std::vector<singleWord> >&, 
            std::vector<std::vector<singleWord> >&, 
            std::unordered_map<string, int>&, 
            std::vector<string>&);

void readDataset(std::string, std::vector<std::vector<singleWord> >&, 
				 std::vector<std::vector<singleWord> >&, std::unordered_map<string, int> &, std::vector<string>&);


void readLine(std::vector<string> &);


void resolutioner(const std::vector<std::vector<singleWord> >&, std::vector<std::vector<int> >&, std::vector<std::vector<int> >&, std::unordered_map<string, int>&);

