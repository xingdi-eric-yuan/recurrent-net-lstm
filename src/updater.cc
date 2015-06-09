#include "updater.h"

using namespace std;

// softmax updater
softmaxUpdater::softmaxUpdater(Smr &smr){
    init(smr);
}

softmaxUpdater::~softmaxUpdater(){

    velocity_W_left.release();
    velocity_W_right.release();
    second_derivative_W_left.release();
    second_derivative_W_right.release();
    learning_rate.release();
}

void softmaxUpdater::init(Smr &smr){
    velocity_W_left = Mat::zeros(smr.W_left.size(), CV_64FC1);
    second_derivative_W_left = Mat::zeros(smr.W_left.size(), CV_64FC1);
    velocity_W_right = Mat::zeros(smr.W_right.size(), CV_64FC1);
    second_derivative_W_right = Mat::zeros(smr.W_right.size(), CV_64FC1);
    iter = 0;
    mu = 1e-2;
    softmaxUpdater::setMomentum();
}

void softmaxUpdater::setMomentum(){
    if(iter <= 30){
        momentum_W = momentum_w_init;
        momentum_second_derivative = momentum_d2_init;
    }else{
        momentum_W = momentum_w_adjust;
        momentum_second_derivative = momentum_d2_adjust;
    }
}

void softmaxUpdater::update(Smr &smr, int iter_num){
    iter = iter_num;
    if(iter > 30) softmaxUpdater::setMomentum();
    second_derivative_W_left = momentum_second_derivative * second_derivative_W_left + (1.0 - momentum_second_derivative) * smr.Wd2_left;
    second_derivative_W_right = momentum_second_derivative * second_derivative_W_right + (1.0 - momentum_second_derivative) * smr.Wd2_right;
    learning_rate = smr.lr_W / (second_derivative_W_left + mu);
    velocity_W_left = velocity_W_left * momentum_W + (1.0 - momentum_W) * smr.Wgrad_left.mul(learning_rate);
    smr.W_left -= velocity_W_left;
    learning_rate = smr.lr_W / (second_derivative_W_right + mu);
    velocity_W_right = velocity_W_right * momentum_W + (1.0 - momentum_W) * smr.Wgrad_right.mul(learning_rate);
    smr.W_right -= velocity_W_right;
}

// LSTM layer updater
LSTMLayerUpdater::LSTMLayerUpdater(std::vector<LSTMl> &HiddenLayers){
    init(HiddenLayers);
}

LSTMLayerUpdater::~LSTMLayerUpdater(){

    velocity_W_input_left.clear();
    std::vector<Mat>().swap(velocity_W_input_left);
    velocity_W_forget_left.clear();
    std::vector<Mat>().swap(velocity_W_forget_left);
    velocity_W_cell_left.clear();
    std::vector<Mat>().swap(velocity_W_cell_left);
    velocity_W_output_left.clear();
    std::vector<Mat>().swap(velocity_W_output_left);

    velocity_U_input_left.clear();
    std::vector<Mat>().swap(velocity_U_input_left);
    velocity_U_forget_left.clear();
    std::vector<Mat>().swap(velocity_U_forget_left);
    velocity_U_cell_left.clear();
    std::vector<Mat>().swap(velocity_U_cell_left);
    velocity_U_output_left.clear();
    std::vector<Mat>().swap(velocity_U_output_left);

    velocity_V_input_left.clear();
    std::vector<diagonalMatrix*>().swap(velocity_V_input_left);
    velocity_V_forget_left.clear();
    std::vector<diagonalMatrix*>().swap(velocity_V_forget_left);
    velocity_V_output_left.clear();
    std::vector<diagonalMatrix*>().swap(velocity_V_output_left);
    
    second_derivative_W_input_left.clear();
    std::vector<Mat>().swap(second_derivative_W_input_left);
    second_derivative_W_forget_left.clear();
    std::vector<Mat>().swap(second_derivative_W_forget_left);
    second_derivative_W_cell_left.clear();
    std::vector<Mat>().swap(second_derivative_W_cell_left);
    second_derivative_W_output_left.clear();
    std::vector<Mat>().swap(second_derivative_W_output_left);

    second_derivative_U_input_left.clear();
    std::vector<Mat>().swap(second_derivative_U_input_left);
    second_derivative_U_forget_left.clear();
    std::vector<Mat>().swap(second_derivative_U_forget_left);
    second_derivative_U_cell_left.clear();
    std::vector<Mat>().swap(second_derivative_U_cell_left);
    second_derivative_U_output_left.clear();
    std::vector<Mat>().swap(second_derivative_U_output_left);

    second_derivative_V_input_left.clear();
    std::vector<diagonalMatrix*>().swap(second_derivative_V_input_left);
    second_derivative_V_forget_left.clear();
    std::vector<diagonalMatrix*>().swap(second_derivative_V_forget_left);
    second_derivative_V_output_left.clear();
    std::vector<diagonalMatrix*>().swap(second_derivative_V_output_left);

    velocity_W_input_right.clear();
    std::vector<Mat>().swap(velocity_W_input_right);
    velocity_W_forget_right.clear();
    std::vector<Mat>().swap(velocity_W_forget_right);
    velocity_W_cell_right.clear();
    std::vector<Mat>().swap(velocity_W_cell_right);
    velocity_W_output_right.clear();
    std::vector<Mat>().swap(velocity_W_output_right);

    velocity_U_input_right.clear();
    std::vector<Mat>().swap(velocity_U_input_right);
    velocity_U_forget_right.clear();
    std::vector<Mat>().swap(velocity_U_forget_right);
    velocity_U_cell_right.clear();
    std::vector<Mat>().swap(velocity_U_cell_right);
    velocity_U_output_right.clear();
    std::vector<Mat>().swap(velocity_U_output_right);

    velocity_V_input_right.clear();
    std::vector<diagonalMatrix*>().swap(velocity_V_input_right);
    velocity_V_forget_right.clear();
    std::vector<diagonalMatrix*>().swap(velocity_V_forget_right);
    velocity_V_output_right.clear();
    std::vector<diagonalMatrix*>().swap(velocity_V_output_right);
    
    second_derivative_W_input_right.clear();
    std::vector<Mat>().swap(second_derivative_W_input_right);
    second_derivative_W_forget_right.clear();
    std::vector<Mat>().swap(second_derivative_W_forget_right);
    second_derivative_W_cell_right.clear();
    std::vector<Mat>().swap(second_derivative_W_cell_right);
    second_derivative_W_output_right.clear();
    std::vector<Mat>().swap(second_derivative_W_output_right);

    second_derivative_U_input_right.clear();
    std::vector<Mat>().swap(second_derivative_U_input_right);
    second_derivative_U_forget_right.clear();
    std::vector<Mat>().swap(second_derivative_U_forget_right);
    second_derivative_U_cell_right.clear();
    std::vector<Mat>().swap(second_derivative_U_cell_right);
    second_derivative_U_output_right.clear();
    std::vector<Mat>().swap(second_derivative_U_output_right);

    second_derivative_V_input_right.clear();
    std::vector<diagonalMatrix*>().swap(second_derivative_V_input_right);
    second_derivative_V_forget_right.clear();
    std::vector<diagonalMatrix*>().swap(second_derivative_V_forget_right);
    second_derivative_V_output_right.clear();
    std::vector<diagonalMatrix*>().swap(second_derivative_V_output_right);

    learning_rate_W.release();
    learning_rate_U.release();
}

void LSTMLayerUpdater::init(std::vector<LSTMl> &HiddenLayers){

    for(int i = 0; i < HiddenLayers.size(); ++i){
        Mat tmpW = Mat::zeros(HiddenLayers[i].W_input_left.size(), CV_64FC1);
        Mat tmpU = Mat::zeros(HiddenLayers[i].U_input_right.size(), CV_64FC1);
        diagonalMatrix *tmpdm;

        velocity_W_input_left.push_back(tmpW);
        velocity_W_forget_left.push_back(tmpW);
        velocity_W_cell_left.push_back(tmpW);
        velocity_W_output_left.push_back(tmpW);
        
        velocity_U_input_left.push_back(tmpU);
        velocity_U_forget_left.push_back(tmpU);
        velocity_U_cell_left.push_back(tmpU);
        velocity_U_output_left.push_back(tmpU);

        tmpdm = new diagonalMatrix(HiddenLayers[i].V_input_left -> diagonal.rows);
        velocity_V_input_left.push_back(tmpdm);
        tmpdm = new diagonalMatrix(HiddenLayers[i].V_input_left -> diagonal.rows);
        velocity_V_forget_left.push_back(tmpdm);
        tmpdm = new diagonalMatrix(HiddenLayers[i].V_input_left -> diagonal.rows);
        velocity_V_output_left.push_back(tmpdm);

        second_derivative_W_input_left.push_back(tmpW);
        second_derivative_W_forget_left.push_back(tmpW);
        second_derivative_W_cell_left.push_back(tmpW);
        second_derivative_W_output_left.push_back(tmpW);
        
        second_derivative_U_input_left.push_back(tmpU);
        second_derivative_U_forget_left.push_back(tmpU);
        second_derivative_U_cell_left.push_back(tmpU);
        second_derivative_U_output_left.push_back(tmpU);

        tmpdm = new diagonalMatrix(HiddenLayers[i].V_input_left -> diagonal.rows);
        second_derivative_V_input_left.push_back(tmpdm);
        tmpdm = new diagonalMatrix(HiddenLayers[i].V_input_left -> diagonal.rows);
        second_derivative_V_forget_left.push_back(tmpdm);
        tmpdm = new diagonalMatrix(HiddenLayers[i].V_input_left -> diagonal.rows);
        second_derivative_V_output_left.push_back(tmpdm);

        velocity_W_input_right.push_back(tmpW);
        velocity_W_forget_right.push_back(tmpW);
        velocity_W_cell_right.push_back(tmpW);
        velocity_W_output_right.push_back(tmpW);
        
        velocity_U_input_right.push_back(tmpU);
        velocity_U_forget_right.push_back(tmpU);
        velocity_U_cell_right.push_back(tmpU);
        velocity_U_output_right.push_back(tmpU);

        tmpdm = new diagonalMatrix(HiddenLayers[i].V_input_right -> diagonal.rows);
        velocity_V_input_right.push_back(tmpdm);
        tmpdm = new diagonalMatrix(HiddenLayers[i].V_input_right -> diagonal.rows);
        velocity_V_forget_right.push_back(tmpdm);
        tmpdm = new diagonalMatrix(HiddenLayers[i].V_input_right -> diagonal.rows);
        velocity_V_output_right.push_back(tmpdm);

        second_derivative_W_input_right.push_back(tmpW);
        second_derivative_W_forget_right.push_back(tmpW);
        second_derivative_W_cell_right.push_back(tmpW);
        second_derivative_W_output_right.push_back(tmpW);
        
        second_derivative_U_input_right.push_back(tmpU);
        second_derivative_U_forget_right.push_back(tmpU);
        second_derivative_U_cell_right.push_back(tmpU);
        second_derivative_U_output_right.push_back(tmpU);

        tmpdm = new diagonalMatrix(HiddenLayers[i].V_input_right -> diagonal.rows);
        second_derivative_V_input_right.push_back(tmpdm);
        tmpdm = new diagonalMatrix(HiddenLayers[i].V_input_right -> diagonal.rows);
        second_derivative_V_forget_right.push_back(tmpdm);
        tmpdm = new diagonalMatrix(HiddenLayers[i].V_input_right -> diagonal.rows);
        second_derivative_V_output_right.push_back(tmpdm);
    }
    iter = 0;
    mu = 1e-2;
    LSTMLayerUpdater::setMomentum();
}

void LSTMLayerUpdater::setMomentum(){
    if(iter <= 30){
        momentum_W = momentum_w_init;
        momentum_second_derivative = momentum_d2_init;
    }else{
        momentum_W = momentum_w_adjust;
        momentum_second_derivative = momentum_d2_adjust;
    }
}

void LSTMLayerUpdater::update(std::vector<LSTMl> &HiddenLayers, int iter_num){
    iter = iter_num;
    if(iter == 30) LSTMLayerUpdater::setMomentum();
    for(int i = 0; i < HiddenLayers.size(); ++i){
        //cout<<endl<<endl<<endl<<" "<<HiddenLayers[i].Wgrad_output<<endl;

        second_derivative_W_input_left[i] = momentum_second_derivative * second_derivative_W_input_left[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Wd2_input_left;
        second_derivative_W_forget_left[i] = momentum_second_derivative * second_derivative_W_forget_left[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Wd2_forget_left;
        second_derivative_W_cell_left[i] = momentum_second_derivative * second_derivative_W_cell_left[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Wd2_cell_left;
        second_derivative_W_output_left[i] = momentum_second_derivative * second_derivative_W_forget_left[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Wd2_output_left;
        
        second_derivative_U_input_left[i] = momentum_second_derivative * second_derivative_U_input_left[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Ud2_input_left;
        second_derivative_U_forget_left[i] = momentum_second_derivative * second_derivative_U_forget_left[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Ud2_forget_left;
        second_derivative_U_cell_left[i] = momentum_second_derivative * second_derivative_U_cell_left[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Ud2_cell_left;
        second_derivative_U_output_left[i] = momentum_second_derivative * second_derivative_U_forget_left[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Ud2_output_left;
        
        second_derivative_V_input_left[i] -> full = momentum_second_derivative * second_derivative_V_input_left[i] -> full + (1.0 - momentum_second_derivative) * HiddenLayers[i].Vd2_input_left -> full;
        second_derivative_V_input_left[i] -> update(UPDATE_FROM_FULL);
        second_derivative_V_forget_left[i] -> full = momentum_second_derivative * second_derivative_V_forget_left[i] -> full + (1.0 - momentum_second_derivative) * HiddenLayers[i].Vd2_forget_left -> full;
        second_derivative_V_forget_left[i] -> update(UPDATE_FROM_FULL);
        second_derivative_V_output_left[i] -> full = momentum_second_derivative * second_derivative_V_output_left[i] -> full + (1.0 - momentum_second_derivative) * HiddenLayers[i].Vd2_output_left -> full;
        second_derivative_V_output_left[i] -> update(UPDATE_FROM_FULL);
        
        learning_rate_W = HiddenLayers[i].lr_W / (second_derivative_W_input_left[i] + mu);
        learning_rate_U = HiddenLayers[i].lr_U / (second_derivative_U_input_left[i] + mu);
        learning_rate_V = HiddenLayers[i].lr_V / (second_derivative_V_input_left[i] -> full + mu);
        velocity_W_input_left[i] = velocity_W_input_left[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Wgrad_input_left.mul(learning_rate_W);
        velocity_U_input_left[i] = velocity_U_input_left[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Ugrad_input_left.mul(learning_rate_U);
        
        velocity_V_input_left[i] -> full = velocity_V_input_left[i] -> full * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Vgrad_input_left -> full.mul(learning_rate_V);
        velocity_V_input_left[i] -> update(UPDATE_FROM_FULL);
        
        learning_rate_W = HiddenLayers[i].lr_W / (second_derivative_W_forget_left[i] + mu);
        learning_rate_U = HiddenLayers[i].lr_U / (second_derivative_U_forget_left[i] + mu);
        learning_rate_V = HiddenLayers[i].lr_V / (second_derivative_V_forget_left[i] -> full + mu);
        velocity_W_forget_left[i] = velocity_W_forget_left[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Wgrad_forget_left.mul(learning_rate_W);
        velocity_U_forget_left[i] = velocity_U_forget_left[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Ugrad_forget_left.mul(learning_rate_U);
        
        velocity_V_forget_left[i] -> full = velocity_V_forget_left[i] -> full * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Vgrad_forget_left -> full.mul(learning_rate_V);
        velocity_V_forget_left[i] -> update(UPDATE_FROM_FULL);
        
        learning_rate_W = HiddenLayers[i].lr_W / (second_derivative_W_output_left[i] + mu);
        learning_rate_U = HiddenLayers[i].lr_U / (second_derivative_U_output_left[i] + mu);
        learning_rate_V = HiddenLayers[i].lr_V / (second_derivative_V_output_left[i] -> full + mu);
        velocity_W_output_left[i] = velocity_W_output_left[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Wgrad_output_left.mul(learning_rate_W);
        velocity_U_output_left[i] = velocity_U_output_left[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Ugrad_output_left.mul(learning_rate_U);
        
        velocity_V_output_left[i] -> full = velocity_V_output_left[i] -> full * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Vgrad_output_left -> full.mul(learning_rate_V);
        velocity_V_output_left[i] -> update(UPDATE_FROM_FULL);
        
        learning_rate_W = HiddenLayers[i].lr_W / (second_derivative_W_cell_left[i] + mu);
        learning_rate_U = HiddenLayers[i].lr_U / (second_derivative_U_cell_left[i] + mu);
        velocity_W_cell_left[i] = velocity_W_cell_left[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Wgrad_cell_left.mul(learning_rate_W);
        velocity_U_cell_left[i] = velocity_U_cell_left[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Ugrad_cell_left.mul(learning_rate_U);
        
        HiddenLayers[i].W_input_left -= velocity_W_input_left[i];
        HiddenLayers[i].W_forget_left -= velocity_W_forget_left[i];
        HiddenLayers[i].W_cell_left -= velocity_W_cell_left[i];
        HiddenLayers[i].W_output_left -= velocity_W_output_left[i];

        HiddenLayers[i].U_input_left -= velocity_U_input_left[i];
        HiddenLayers[i].U_forget_left -= velocity_U_forget_left[i];
        HiddenLayers[i].U_cell_left -= velocity_U_cell_left[i];
        HiddenLayers[i].U_output_left -= velocity_U_output_left[i];

        HiddenLayers[i].V_input_left -> full -= velocity_V_input_left[i] -> full;
        HiddenLayers[i].V_forget_left -> full -= velocity_V_forget_left[i] -> full;
        HiddenLayers[i].V_output_left -> full -= velocity_V_output_left[i] -> full;

        HiddenLayers[i].V_input_left -> update(UPDATE_FROM_FULL);
        HiddenLayers[i].V_forget_left -> update(UPDATE_FROM_FULL);
        HiddenLayers[i].V_output_left -> update(UPDATE_FROM_FULL);

        
        second_derivative_W_input_right[i] = momentum_second_derivative * second_derivative_W_input_right[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Wd2_input_right;
        second_derivative_W_forget_right[i] = momentum_second_derivative * second_derivative_W_forget_right[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Wd2_forget_right;
        second_derivative_W_cell_right[i] = momentum_second_derivative * second_derivative_W_cell_right[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Wd2_cell_right;
        second_derivative_W_output_right[i] = momentum_second_derivative * second_derivative_W_forget_right[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Wd2_output_right;
        
        second_derivative_U_input_right[i] = momentum_second_derivative * second_derivative_U_input_right[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Ud2_input_right;
        second_derivative_U_forget_right[i] = momentum_second_derivative * second_derivative_U_forget_right[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Ud2_forget_right;
        second_derivative_U_cell_right[i] = momentum_second_derivative * second_derivative_U_cell_right[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Ud2_cell_right;
        second_derivative_U_output_right[i] = momentum_second_derivative * second_derivative_U_forget_right[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Ud2_output_right;
        
        second_derivative_V_input_right[i] -> full = momentum_second_derivative * second_derivative_V_input_right[i] -> full + (1.0 - momentum_second_derivative) * HiddenLayers[i].Vd2_input_right -> full;
        second_derivative_V_input_right[i] -> update(UPDATE_FROM_FULL);
        second_derivative_V_forget_right[i] -> full = momentum_second_derivative * second_derivative_V_forget_right[i] -> full + (1.0 - momentum_second_derivative) * HiddenLayers[i].Vd2_forget_right -> full;
        second_derivative_V_forget_right[i] -> update(UPDATE_FROM_FULL);
        second_derivative_V_output_right[i] -> full = momentum_second_derivative * second_derivative_V_output_right[i] -> full + (1.0 - momentum_second_derivative) * HiddenLayers[i].Vd2_output_right -> full;
        second_derivative_V_output_right[i] -> update(UPDATE_FROM_FULL);
        
        learning_rate_W = HiddenLayers[i].lr_W / (second_derivative_W_input_right[i] + mu);
        learning_rate_U = HiddenLayers[i].lr_U / (second_derivative_U_input_right[i] + mu);
        learning_rate_V = HiddenLayers[i].lr_V / (second_derivative_V_input_right[i] -> full + mu);
        velocity_W_input_right[i] = velocity_W_input_right[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Wgrad_input_right.mul(learning_rate_W);
        velocity_U_input_right[i] = velocity_U_input_right[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Ugrad_input_right.mul(learning_rate_U);
        
        velocity_V_input_right[i] -> full = velocity_V_input_right[i] -> full * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Vgrad_input_right -> full.mul(learning_rate_V);
        velocity_V_input_right[i] -> update(UPDATE_FROM_FULL);
        
        learning_rate_W = HiddenLayers[i].lr_W / (second_derivative_W_forget_right[i] + mu);
        learning_rate_U = HiddenLayers[i].lr_U / (second_derivative_U_forget_right[i] + mu);
        learning_rate_V = HiddenLayers[i].lr_V / (second_derivative_V_forget_right[i] -> full + mu);
        velocity_W_forget_right[i] = velocity_W_forget_right[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Wgrad_forget_right.mul(learning_rate_W);
        velocity_U_forget_right[i] = velocity_U_forget_right[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Ugrad_forget_right.mul(learning_rate_U);
        
        velocity_V_forget_right[i] -> full = velocity_V_forget_right[i] -> full * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Vgrad_forget_right -> full.mul(learning_rate_V);
        velocity_V_forget_right[i] -> update(UPDATE_FROM_FULL);
        
        learning_rate_W = HiddenLayers[i].lr_W / (second_derivative_W_output_right[i] + mu);
        learning_rate_U = HiddenLayers[i].lr_U / (second_derivative_U_output_right[i] + mu);
        learning_rate_V = HiddenLayers[i].lr_V / (second_derivative_V_output_right[i] -> full + mu);
        velocity_W_output_right[i] = velocity_W_output_right[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Wgrad_output_right.mul(learning_rate_W);
        velocity_U_output_right[i] = velocity_U_output_right[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Ugrad_output_right.mul(learning_rate_U);
        
        velocity_V_output_right[i] -> full = velocity_V_output_right[i] -> full * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Vgrad_output_right -> full.mul(learning_rate_V);
        velocity_V_output_right[i] -> update(UPDATE_FROM_FULL);
        
        learning_rate_W = HiddenLayers[i].lr_W / (second_derivative_W_cell_right[i] + mu);
        learning_rate_U = HiddenLayers[i].lr_U / (second_derivative_U_cell_right[i] + mu);
        velocity_W_cell_right[i] = velocity_W_cell_right[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Wgrad_cell_right.mul(learning_rate_W);
        velocity_U_cell_right[i] = velocity_U_cell_right[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Ugrad_cell_right.mul(learning_rate_U);
        
        HiddenLayers[i].W_input_right -= velocity_W_input_right[i];
        HiddenLayers[i].W_forget_right -= velocity_W_forget_right[i];
        HiddenLayers[i].W_cell_right -= velocity_W_cell_right[i];
        HiddenLayers[i].W_output_right -= velocity_W_output_right[i];

        HiddenLayers[i].U_input_right -= velocity_U_input_right[i];
        HiddenLayers[i].U_forget_right -= velocity_U_forget_right[i];
        HiddenLayers[i].U_cell_right -= velocity_U_cell_right[i];
        HiddenLayers[i].U_output_right -= velocity_U_output_right[i];

        HiddenLayers[i].V_input_right -> full -= velocity_V_input_right[i] -> full;
        HiddenLayers[i].V_forget_right -> full -= velocity_V_forget_right[i] -> full;
        HiddenLayers[i].V_output_right -> full -= velocity_V_output_right[i] -> full;

        HiddenLayers[i].V_input_right -> update(UPDATE_FROM_FULL);
        HiddenLayers[i].V_forget_right -> update(UPDATE_FROM_FULL);
        HiddenLayers[i].V_output_right -> update(UPDATE_FROM_FULL);
    }
}
//*/


