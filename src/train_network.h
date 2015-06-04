#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

void trainNetwork(const std::vector<std::vector<int> > &, std::vector<std::vector<int> > &, std::vector<LSTMl> &, Smr &, 
				  const std::vector<std::vector<int> > &, std::vector<std::vector<int> > &, std::vector<string>&, 
				  unordered_map<string, Mat>&);