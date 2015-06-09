#include "weight_init.h"

using namespace cv;
using namespace std;
void
weightRandomInit(LSTMl &ntw, int inputsize, int hiddensize){
    double epsilon = 0.12;
    //// LEFT
    // U
    ntw.U_input_left = Mat::ones(hiddensize, inputsize, CV_64FC1);
    randu(ntw.U_input_left, Scalar(-1.0), Scalar(1.0));
    ntw.U_input_left = ntw.U_input_left * epsilon;
    ntw.Ugrad_input_left = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.Ud2_input_left = Mat::zeros(ntw.U_input_left.size(), CV_64FC1);

    ntw.U_forget_left = Mat::ones(hiddensize, inputsize, CV_64FC1);
    randu(ntw.U_forget_left, Scalar(-1.0), Scalar(1.0));
    ntw.U_forget_left = ntw.U_forget_left * epsilon;
    ntw.Ugrad_forget_left = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.Ud2_forget_left = Mat::zeros(ntw.U_forget_left.size(), CV_64FC1);

    ntw.U_output_left = Mat::ones(hiddensize, inputsize, CV_64FC1);
    randu(ntw.U_output_left, Scalar(-1.0), Scalar(1.0));
    ntw.U_output_left = ntw.U_output_left * epsilon;
    ntw.Ugrad_output_left = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.Ud2_output_left = Mat::zeros(ntw.U_output_left.size(), CV_64FC1);

    ntw.U_cell_left = Mat::ones(hiddensize, inputsize, CV_64FC1);
    randu(ntw.U_cell_left, Scalar(-1.0), Scalar(1.0));
    ntw.U_cell_left = ntw.U_cell_left * epsilon;
    ntw.Ugrad_cell_left = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.Ud2_cell_left = Mat::zeros(ntw.U_cell_left.size(), CV_64FC1);

    // W
    ntw.W_input_left = Mat::ones(hiddensize, hiddensize, CV_64FC1);
    randu(ntw.W_input_left, Scalar(-1.0), Scalar(1.0));
    ntw.W_input_left = ntw.W_input_left * epsilon;
    ntw.Wgrad_input_left = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
    ntw.Wd2_input_left = Mat::zeros(ntw.W_input_left.size(), CV_64FC1);

    ntw.W_forget_left = Mat::ones(hiddensize, hiddensize, CV_64FC1);
    randu(ntw.W_forget_left, Scalar(-1.0), Scalar(1.0));
    ntw.W_forget_left = ntw.W_forget_left * epsilon;
    ntw.Wgrad_forget_left = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
    ntw.Wd2_forget_left = Mat::zeros(ntw.W_forget_left.size(), CV_64FC1);

    ntw.W_output_left = Mat::ones(hiddensize, hiddensize, CV_64FC1);
    randu(ntw.W_output_left, Scalar(-1.0), Scalar(1.0));
    ntw.W_output_left = ntw.W_output_left * epsilon;
    ntw.Wgrad_output_left = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
    ntw.Wd2_output_left = Mat::zeros(ntw.W_output_left.size(), CV_64FC1);

    ntw.W_cell_left = Mat::ones(hiddensize, hiddensize, CV_64FC1);
    randu(ntw.W_cell_left, Scalar(-1.0), Scalar(1.0));
    ntw.W_cell_left = ntw.W_cell_left * epsilon;
    ntw.Wgrad_cell_left = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
    ntw.Wd2_cell_left = Mat::zeros(ntw.W_cell_left.size(), CV_64FC1);
    // V
    ntw.V_input_left = new diagonalMatrix();
    ntw.Vgrad_input_left = new diagonalMatrix();
    ntw.Vd2_input_left = new diagonalMatrix();
    ntw.V_forget_left = new diagonalMatrix();
    ntw.Vgrad_forget_left = new diagonalMatrix();
    ntw.Vd2_forget_left = new diagonalMatrix();
    ntw.V_output_left = new diagonalMatrix();
    ntw.Vgrad_output_left = new diagonalMatrix();
    ntw.Vd2_output_left = new diagonalMatrix();
    
    ntw.V_input_left -> randomInit(hiddensize, epsilon);
    ntw.Vgrad_input_left -> init(hiddensize);
    ntw.Vd2_input_left -> init(hiddensize);

    ntw.V_forget_left -> randomInit(hiddensize, epsilon);
    ntw.Vgrad_forget_left -> init(hiddensize);
    ntw.Vd2_forget_left -> init(hiddensize);

    ntw.V_output_left -> randomInit(hiddensize, epsilon);
    ntw.Vgrad_output_left -> init(hiddensize);
    ntw.Vd2_output_left -> init(hiddensize);
    //// RIGHT
    // U
    ntw.U_input_right = Mat::ones(hiddensize, inputsize, CV_64FC1);
    randu(ntw.U_input_right, Scalar(-1.0), Scalar(1.0));
    ntw.U_input_right = ntw.U_input_right * epsilon;
    ntw.Ugrad_input_right = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.Ud2_input_right = Mat::zeros(ntw.U_input_right.size(), CV_64FC1);

    ntw.U_forget_right = Mat::ones(hiddensize, inputsize, CV_64FC1);
    randu(ntw.U_forget_right, Scalar(-1.0), Scalar(1.0));
    ntw.U_forget_right = ntw.U_forget_right * epsilon;
    ntw.Ugrad_forget_right = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.Ud2_forget_right = Mat::zeros(ntw.U_forget_right.size(), CV_64FC1);

    ntw.U_output_right = Mat::ones(hiddensize, inputsize, CV_64FC1);
    randu(ntw.U_output_right, Scalar(-1.0), Scalar(1.0));
    ntw.U_output_right = ntw.U_output_right * epsilon;
    ntw.Ugrad_output_right = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.Ud2_output_right = Mat::zeros(ntw.U_output_right.size(), CV_64FC1);

    ntw.U_cell_right = Mat::ones(hiddensize, inputsize, CV_64FC1);
    randu(ntw.U_cell_right, Scalar(-1.0), Scalar(1.0));
    ntw.U_cell_right = ntw.U_cell_right * epsilon;
    ntw.Ugrad_cell_right = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.Ud2_cell_right = Mat::zeros(ntw.U_cell_right.size(), CV_64FC1);

    // W
    ntw.W_input_right = Mat::ones(hiddensize, hiddensize, CV_64FC1);
    randu(ntw.W_input_right, Scalar(-1.0), Scalar(1.0));
    ntw.W_input_right = ntw.W_input_right * epsilon;
    ntw.Wgrad_input_right = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
    ntw.Wd2_input_right = Mat::zeros(ntw.W_input_right.size(), CV_64FC1);

    ntw.W_forget_right = Mat::ones(hiddensize, hiddensize, CV_64FC1);
    randu(ntw.W_forget_right, Scalar(-1.0), Scalar(1.0));
    ntw.W_forget_right = ntw.W_forget_right * epsilon;
    ntw.Wgrad_forget_right = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
    ntw.Wd2_forget_right = Mat::zeros(ntw.W_forget_right.size(), CV_64FC1);

    ntw.W_output_right = Mat::ones(hiddensize, hiddensize, CV_64FC1);
    randu(ntw.W_output_right, Scalar(-1.0), Scalar(1.0));
    ntw.W_output_right = ntw.W_output_right * epsilon;
    ntw.Wgrad_output_right = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
    ntw.Wd2_output_right = Mat::zeros(ntw.W_output_right.size(), CV_64FC1);

    ntw.W_cell_right = Mat::ones(hiddensize, hiddensize, CV_64FC1);
    randu(ntw.W_cell_right, Scalar(-1.0), Scalar(1.0));
    ntw.W_cell_right = ntw.W_cell_right * epsilon;
    ntw.Wgrad_cell_right = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
    ntw.Wd2_cell_right = Mat::zeros(ntw.W_cell_right.size(), CV_64FC1);
    // V
    ntw.V_input_right = new diagonalMatrix();
    ntw.Vgrad_input_right = new diagonalMatrix();
    ntw.Vd2_input_right = new diagonalMatrix();
    ntw.V_forget_right = new diagonalMatrix();
    ntw.Vgrad_forget_right = new diagonalMatrix();
    ntw.Vd2_forget_right = new diagonalMatrix();
    ntw.V_output_right = new diagonalMatrix();
    ntw.Vgrad_output_right = new diagonalMatrix();
    ntw.Vd2_output_right = new diagonalMatrix();
    
    ntw.V_input_right -> randomInit(hiddensize, epsilon);
    ntw.Vgrad_input_right -> init(hiddensize);
    ntw.Vd2_input_right -> init(hiddensize);

    ntw.V_forget_right -> randomInit(hiddensize, epsilon);
    ntw.Vgrad_forget_right -> init(hiddensize);
    ntw.Vd2_forget_right -> init(hiddensize);

    ntw.V_output_right -> randomInit(hiddensize, epsilon);
    ntw.Vgrad_output_right -> init(hiddensize);
    ntw.Vd2_output_right -> init(hiddensize);

    ntw.lr_U = lrate_w;
    ntw.lr_W = lrate_w;
    ntw.lr_V = lrate_w;
}

void 
weightRandomInit(Smr &smr, int nclasses, int nfeatures){
    double epsilon = 0.12;
    smr.W_left = Mat::ones(nclasses, nfeatures, CV_64FC1);
    randu(smr.W_left, Scalar(-1.0), Scalar(1.0));
    smr.W_left = smr.W_left * epsilon;
    smr.cost = 0.0;
    smr.Wgrad_left = Mat::zeros(nclasses, nfeatures, CV_64FC1);
    smr.Wd2_left = Mat::zeros(smr.W_left.size(), CV_64FC1);

    smr.W_right = Mat::ones(nclasses, nfeatures, CV_64FC1);
    randu(smr.W_right, Scalar(-1.0), Scalar(1.0));
    smr.W_right = smr.W_right * epsilon;
    smr.cost = 0.0;
    smr.Wgrad_right = Mat::zeros(nclasses, nfeatures, CV_64FC1);
    smr.Wd2_right = Mat::zeros(smr.W_right.size(), CV_64FC1);
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


