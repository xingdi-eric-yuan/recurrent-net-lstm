#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;
void gradient_checking(std::vector<Mat>&, Mat&, std::vector<LSTMl>&, Smr&, Mat&, Mat*);
void gradientChecking_SoftmaxLayer(std::vector<LSTMl> &, Smr &, std::vector<Mat> &, Mat&);
void gradientChecking_LSTMLayer (std::vector<LSTMl> &, Smr &, std::vector<Mat> &, Mat&, int);
