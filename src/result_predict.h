#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

Mat resultPredict(std::vector<Mat> &, std::vector<LSTMl> &, Smr &);

void testNetwork(const std::vector<std::vector<int> > &, std::vector<std::vector<int> > &, std::vector<LSTMl> &, Smr &, std::vector<string> &);
void testNetwork(const std::vector<std::vector<int> > &, std::vector<std::vector<int> > &, std::vector<LSTMl> &, Smr &, std::vector<string> &, std::unordered_map<string, Mat>&);
