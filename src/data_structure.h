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
    //// LEFT
    // weight between current time t with previous time t-1
    Mat W_input_left;
    Mat W_forget_left;
    Mat W_output_left;
    Mat W_cell_left;
    // weight between hidden layer with previous layer
    Mat U_input_left;
    Mat U_forget_left;
    Mat U_output_left;
    Mat U_cell_left;
    // peephole weights
    diagonalMatrix *V_input_left;
    diagonalMatrix *V_forget_left;
    diagonalMatrix *V_output_left;
    // 1st order derivative
    Mat Wgrad_input_left;
    Mat Wgrad_forget_left;
    Mat Wgrad_output_left;
    Mat Wgrad_cell_left;

    Mat Ugrad_input_left;
    Mat Ugrad_forget_left;
    Mat Ugrad_output_left;
    Mat Ugrad_cell_left;
        
    diagonalMatrix *Vgrad_input_left;
    diagonalMatrix *Vgrad_forget_left;
    diagonalMatrix *Vgrad_output_left;
    // 2nd order derivative
    Mat Wd2_input_left;
    Mat Wd2_forget_left;
    Mat Wd2_output_left;
    Mat Wd2_cell_left;

    Mat Ud2_input_left;
    Mat Ud2_forget_left;
    Mat Ud2_output_left;
    Mat Ud2_cell_left;
        
    diagonalMatrix *Vd2_input_left;
    diagonalMatrix *Vd2_forget_left;
    diagonalMatrix *Vd2_output_left;
    //// RIGHT
    // weight between current time t with previous time t-1
    Mat W_input_right;
    Mat W_forget_right;
    Mat W_output_right;
    Mat W_cell_right;
    // weight between hidden layer with previous layer
    Mat U_input_right;
    Mat U_forget_right;
    Mat U_output_right;
    Mat U_cell_right;
    // peephole weights
    diagonalMatrix *V_input_right;
    diagonalMatrix *V_forget_right;
    diagonalMatrix *V_output_right;
    // 1st order derivative
    Mat Wgrad_input_right;
    Mat Wgrad_forget_right;
    Mat Wgrad_output_right;
    Mat Wgrad_cell_right;

    Mat Ugrad_input_right;
    Mat Ugrad_forget_right;
    Mat Ugrad_output_right;
    Mat Ugrad_cell_right;
        
    diagonalMatrix *Vgrad_input_right;
    diagonalMatrix *Vgrad_forget_right;
    diagonalMatrix *Vgrad_output_right;
    // 2nd order derivative
    Mat Wd2_input_right;
    Mat Wd2_forget_right;
    Mat Wd2_output_right;
    Mat Wd2_cell_right;

    Mat Ud2_input_right;
    Mat Ud2_forget_right;
    Mat Ud2_output_right;
    Mat Ud2_cell_right;
        
    diagonalMatrix *Vd2_input_right;
    diagonalMatrix *Vd2_forget_right;
    diagonalMatrix *Vd2_output_right;

    double lr_W;
    double lr_U;
    double lr_V;
}LSTMl;

typedef struct SoftmaxRegession{

    Mat W_left;
    Mat Wgrad_left;
    Mat Wd2_left;
    Mat W_right;
    Mat Wgrad_right;
    Mat Wd2_right;
    double cost;
    double lr_W;
}Smr;

///////////////////////////////////
// Config Structures
///////////////////////////////////
///
struct HiddenLayerConfig {
    int NumHiddenNeurons;
    double WeightDecay;
    HiddenLayerConfig(int a, double b) : NumHiddenNeurons(a), WeightDecay(b) {}
};

struct SoftmaxLayerConfig {
    int NumClasses;
    double WeightDecay;
    //SoftmaxLayerConfig(int a, double b) : NumClasses(a), WeightDecay(b) {}
};