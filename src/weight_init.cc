#include "weight_init.h"

using namespace cv;
using namespace std;
void
weightRandomInit(LSTMl &ntw, int inputsize, int hiddensize){
    double epsilon = 0.12;

    // U
    ntw.U_input = Mat::ones(hiddensize, inputsize, CV_64FC1);
    randu(ntw.U_input, Scalar(-1.0), Scalar(1.0));
    ntw.U_input = ntw.U_input * epsilon;
    ntw.Ugrad_input = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.Ud2_input = Mat::zeros(ntw.U_input.size(), CV_64FC1);

    ntw.U_forget = Mat::ones(hiddensize, inputsize, CV_64FC1);
    randu(ntw.U_forget, Scalar(-1.0), Scalar(1.0));
    ntw.U_forget = ntw.U_forget * epsilon;
    ntw.Ugrad_forget = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.Ud2_forget = Mat::zeros(ntw.U_forget.size(), CV_64FC1);

    ntw.U_output = Mat::ones(hiddensize, inputsize, CV_64FC1);
    randu(ntw.U_output, Scalar(-1.0), Scalar(1.0));
    ntw.U_output = ntw.U_output * epsilon;
    ntw.Ugrad_output = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.Ud2_output = Mat::zeros(ntw.U_output.size(), CV_64FC1);

    ntw.U_cell = Mat::ones(hiddensize, inputsize, CV_64FC1);
    randu(ntw.U_cell, Scalar(-1.0), Scalar(1.0));
    ntw.U_cell = ntw.U_cell * epsilon;
    ntw.Ugrad_cell = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.Ud2_cell = Mat::zeros(ntw.U_cell.size(), CV_64FC1);

    // W
    ntw.W_input = Mat::ones(hiddensize, hiddensize, CV_64FC1);
    randu(ntw.W_input, Scalar(-1.0), Scalar(1.0));
    ntw.W_input = ntw.W_input * epsilon;
    ntw.Wgrad_input = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
    ntw.Wd2_input = Mat::zeros(ntw.W_input.size(), CV_64FC1);

    ntw.W_forget = Mat::ones(hiddensize, hiddensize, CV_64FC1);
    randu(ntw.W_forget, Scalar(-1.0), Scalar(1.0));
    ntw.W_forget = ntw.W_forget * epsilon;
    ntw.Wgrad_forget = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
    ntw.Wd2_forget = Mat::zeros(ntw.W_forget.size(), CV_64FC1);

    ntw.W_output = Mat::ones(hiddensize, hiddensize, CV_64FC1);
    randu(ntw.W_output, Scalar(-1.0), Scalar(1.0));
    ntw.W_output = ntw.W_output * epsilon;
    ntw.Wgrad_output = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
    ntw.Wd2_output = Mat::zeros(ntw.W_output.size(), CV_64FC1);

    ntw.W_cell = Mat::ones(hiddensize, hiddensize, CV_64FC1);
    randu(ntw.W_cell, Scalar(-1.0), Scalar(1.0));
    ntw.W_cell = ntw.W_cell * epsilon;
    ntw.Wgrad_cell = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
    ntw.Wd2_cell = Mat::zeros(ntw.W_cell.size(), CV_64FC1);

    // V
    ntw.V_input = new diagonalMatrix();
    ntw.Vgrad_input = new diagonalMatrix();
    ntw.Vd2_input = new diagonalMatrix();
    ntw.V_forget = new diagonalMatrix();
    ntw.Vgrad_forget = new diagonalMatrix();
    ntw.Vd2_forget = new diagonalMatrix();
    ntw.V_output = new diagonalMatrix();
    ntw.Vgrad_output = new diagonalMatrix();
    ntw.Vd2_output = new diagonalMatrix();
    
    ntw.V_input -> randomInit(hiddensize, epsilon);
    ntw.Vgrad_input -> init(hiddensize);
    ntw.Vd2_input -> init(hiddensize);

    ntw.V_forget -> randomInit(hiddensize, epsilon);
    ntw.Vgrad_forget -> init(hiddensize);
    ntw.Vd2_forget -> init(hiddensize);

    ntw.V_output -> randomInit(hiddensize, epsilon);
    ntw.Vgrad_output -> init(hiddensize);
    ntw.Vd2_output -> init(hiddensize);

    ntw.lr_U = lrate_w;
    ntw.lr_W = lrate_w;
    ntw.lr_V = lrate_w;
}

void
weightRandomInit(Rl &ntw, int inputsize, int hiddensize){
    double epsilon = 0.12;
    // weight between hidden layer with previous layer
    ntw.U = Mat::ones(hiddensize, inputsize, CV_64FC1);
    randu(ntw.U, Scalar(-1.0), Scalar(1.0));
    ntw.U = ntw.U * epsilon;
    ntw.Ugrad = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.Ud2 = Mat::zeros(ntw.U.size(), CV_64FC1);
    ntw.lr_U = lrate_w;
    // weight between current time t with previous time t-1
    ntw.W = Mat::ones(hiddensize, hiddensize, CV_64FC1);
    randu(ntw.W, Scalar(-1.0), Scalar(1.0));
    ntw.W = ntw.W * epsilon;
    ntw.Wgrad = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
    ntw.Wd2 = Mat::zeros(ntw.W.size(), CV_64FC1);
    ntw.lr_W = lrate_w;
}

void 
weightRandomInit(Smr &smr, int nclasses, int nfeatures){
    double epsilon = 0.12;
    smr.W = Mat::ones(nclasses, nfeatures, CV_64FC1);
    randu(smr.W, Scalar(-1.0), Scalar(1.0));
    smr.W = smr.W * epsilon;
    smr.cost = 0.0;
    smr.Wgrad = Mat::zeros(nclasses, nfeatures, CV_64FC1);
    smr.Wd2 = Mat::zeros(smr.W.size(), CV_64FC1);
    smr.lr_W = lrate_w;
}

void
rnnInitPrarms(std::vector<LSTMl> &HiddenLayers, Smr &smr){
    
    // Init Hidden layers
    if(hiddenConfig.size() > 0){
        LSTMl tpntw; 
        weightRandomInit(tpntw, word_vec_len, hiddenConfig[0].NumHiddenNeurons);
        HiddenLayers.push_back(tpntw);
        for(int i = 1; i < hiddenConfig.size(); i++){
            LSTMl tpntw2;
            weightRandomInit(tpntw2, hiddenConfig[i - 1].NumHiddenNeurons, hiddenConfig[i].NumHiddenNeurons);
            HiddenLayers.push_back(tpntw2);
        }
    }
    // Init Softmax layer
    if(hiddenConfig.size() == 0){
        weightRandomInit(smr, softmaxConfig.NumClasses, word_vec_len);
    }else{
        weightRandomInit(smr, softmaxConfig.NumClasses, hiddenConfig[hiddenConfig.size() - 1].NumHiddenNeurons);
    }
}


