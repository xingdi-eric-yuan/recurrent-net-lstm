#pragma once
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "diagonal_matrix.h"
#include "data_structure.h"
#include "read_data.h"
#include "helper.h"
#include "cost_gradient.h"
#include "gradient_checking.h"
#include "helper.h"
#include "updater.h"
#include "matrix_maths.h"
#include "weights_IO.h"
#include "train_network.h"
#include "weight_init.h"
#include "result_predict.h"
#include "read_config.h"

// non-linearity
#define NL_SIGMOID 0
#define NL_TANH 1
#define NL_RELU 2

#define IO_NL 0
#define GATE_NL 1

#define UPDATE_FROM_DIAG 0
#define UPDATE_FROM_FULL 1

#define ATD at<double>
#define elif else if

using namespace std;
using namespace cv;

///////////////////////////////////
// General parameters
///////////////////////////////////
extern float training_percent;


extern std::vector<HiddenLayerConfig> hiddenConfig;
extern SoftmaxLayerConfig softmaxConfig;
extern std::vector<int> sample_vec;

///////////////////////////////////
// General parameters
///////////////////////////////////
extern bool is_gradient_checking;
extern bool use_log;
extern int batch_size;
extern int log_iter;
extern bool use_word2vec;

extern int io_non_linearity;
extern int gate_non_linearity;
extern int training_epochs;
extern double lrate_w;
extern double momentum_w_init;
extern double momentum_d2_init;
extern double momentum_w_adjust;
extern double momentum_d2_adjust;
extern int iter_per_epo;
extern int nGram;
extern int word_vec_len;
extern double prev_cost;

