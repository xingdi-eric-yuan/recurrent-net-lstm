#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

void getNetworkCost(std::vector<Mat>&, Mat&, std::vector<Rl>&, Smr&);

bool getNetworkCost(std::vector<Mat>&, Mat&, std::vector<LSTMl>&, Smr&);