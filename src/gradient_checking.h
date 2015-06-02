#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

void gradientChecking_SoftmaxLayer(std::vector<LSTMl> &, Smr &, std::vector<Mat> &, Mat&);
void gradientChecking_RecurrentLayer (std::vector<Rl> &, Smr &, std::vector<Mat> &, Mat&, int);
void gradientChecking_LSTMLayer (std::vector<LSTMl> &, Smr &, std::vector<Mat> &, Mat&, int);
