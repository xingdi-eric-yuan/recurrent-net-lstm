#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;

class diagonalMatrix;
///////////////////////////////////
// mitie Structures
///////////////////////////////////
struct singleWord {
    std::string word;
    int label;
    singleWord(string a, int b) : word(a), label(b){}
};

///////////////////////////////////
// Network Layer Structures
///////////////////////////////////
typedef struct LSTMLayer{
    // weight between current time t with previous time t-1
    Mat W_input;
    Mat W_forget;
    Mat W_output;
    Mat W_cell;
    // weight between hidden layer with previous layer
    Mat U_input;
    Mat U_forget;
    Mat U_output;
    Mat U_cell;
    // peephole weights
    diagonalMatrix *V_input;
    diagonalMatrix *V_forget;
    diagonalMatrix *V_output;
    // 1st order derivative
    Mat Wgrad_input;
    Mat Wgrad_forget;
    Mat Wgrad_output;
    Mat Wgrad_cell;

    Mat Ugrad_input;
    Mat Ugrad_forget;
    Mat Ugrad_output;
    Mat Ugrad_cell;
        
    diagonalMatrix *Vgrad_input;
    diagonalMatrix *Vgrad_forget;
    diagonalMatrix *Vgrad_output;
    // 2nd order derivative
    Mat Wd2_input;
    Mat Wd2_forget;
    Mat Wd2_output;
    Mat Wd2_cell;

    Mat Ud2_input;
    Mat Ud2_forget;
    Mat Ud2_output;
    Mat Ud2_cell;
        
    diagonalMatrix *Vd2_input;
    diagonalMatrix *Vd2_forget;
    diagonalMatrix *Vd2_output;

    double lr_W;
    double lr_U;
    double lr_V;
}LSTMl;


typedef struct RecurrentLayer{
    Mat W;  // weight between current time t with previous time t-1
    Mat U;  // weight between hidden layer with previous layer
    Mat Wgrad;
    Mat Ugrad;
    Mat Wd2;
    Mat Ud2;
    double lr_W;
    double lr_U;
}Rl;

typedef struct SoftmaxRegession{
    Mat W;
    Mat Wgrad;
    double cost;
    Mat Wd2;
    double lr_W;
}Smr;

///////////////////////////////////
// Config Structures
///////////////////////////////////
///
struct HiddenLayerConfig {
    int NumHiddenNeurons;
    double WeightDecay;
    double DropoutRate;
    HiddenLayerConfig(int a, double b, double c) : NumHiddenNeurons(a), WeightDecay(b), DropoutRate(c) {}
};

struct SoftmaxLayerConfig {
    int NumClasses;
    double WeightDecay;
    //SoftmaxLayerConfig(int a, double b) : NumClasses(a), WeightDecay(b) {}
};