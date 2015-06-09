#pragma once
#include "general_settings.h"

using namespace cv;
using namespace std;

void weightRandomInit(LSTMl&, int, int);

void weightRandomInit(Smr&, int, int);

void rnnInitPrarms(std::vector<LSTMl>&, Smr&);