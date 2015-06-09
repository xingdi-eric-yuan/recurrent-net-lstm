#pragma once
#include "general_settings.h"
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;

void save2txt(const Mat&, string, string);

void save2XML(string, string, const std::vector<LSTMl>&, const Smr&, const std::vector<string>&);

void readFromXML(string, std::vector<LSTMl>&, Smr&, std::vector<string>&);
