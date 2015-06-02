#pragma once
#include "general_settings.h"

using namespace std;

// int <==> string
string i2str(int);
int str2i(string);

Mat vec2Mat(const std::vector<int>&);

// label - number look up tables
int label2num(std::string);
std::string num2label(int);

bool isNumber(std::string&);

void removeNumber(std::vector<std::vector<singleWord> >&);

void getWordMap(const std::vector<std::vector<singleWord> >&, std::unordered_map<string, int>&, std::vector<string>&);

void breakString(string , std::vector<string> &);

Mat oneOfN(int, int);

void getSample(const std::vector<std::vector<int> >&, std::vector<Mat>&, const std::vector<std::vector<int> >&, Mat&, std::vector<string> &);

void getDataMat(const std::vector<std::vector<int> >& , std::vector<Mat>& , std::vector<string> &);
void getLabelMat(const std::vector<std::vector<int> >& , Mat&);