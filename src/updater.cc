#include "updater.h"

using namespace std;

// softmax updater
softmaxUpdater::softmaxUpdater(Smr &smr){
    init(smr);
}

softmaxUpdater::~softmaxUpdater(){

    velocity_W.release();
    second_derivative_W.release();
    learning_rate.release();
}

void softmaxUpdater::init(Smr &smr){
    velocity_W = Mat::zeros(smr.W.size(), CV_64FC1);
    second_derivative_W = Mat::zeros(smr.W.size(), CV_64FC1);
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
    second_derivative_W = momentum_second_derivative * second_derivative_W + (1.0 - momentum_second_derivative) * smr.Wd2;
    learning_rate = smr.lr_W / (second_derivative_W + mu);
    velocity_W = velocity_W * momentum_W + (1.0 - momentum_W) * smr.Wgrad.mul(learning_rate);
    smr.W -= velocity_W;
}


// recurrent layer updater
recurrentLayerUpdater::recurrentLayerUpdater(std::vector<Rl> &HiddenLayers){
    init(HiddenLayers);
}

recurrentLayerUpdater::~recurrentLayerUpdater(){

    velocity_W.clear();
    std::vector<Mat>().swap(velocity_W);
    velocity_U.clear();
    std::vector<Mat>().swap(velocity_U);
    second_derivative_W.clear();
    std::vector<Mat>().swap(second_derivative_W);
    second_derivative_U.clear();
    std::vector<Mat>().swap(second_derivative_U);
    learning_rate_W.release();
    learning_rate_U.release();
}

void recurrentLayerUpdater::init(std::vector<Rl> &HiddenLayers){

    for(int i = 0; i < HiddenLayers.size(); ++i){
        Mat tmpW = Mat::zeros(HiddenLayers[i].W.size(), CV_64FC1);
        Mat tmpU = Mat::zeros(HiddenLayers[i].U.size(), CV_64FC1);
        velocity_W.push_back(tmpW);
        velocity_U.push_back(tmpU);
        second_derivative_W.push_back(tmpW);
        second_derivative_U.push_back(tmpU);
    }
    iter = 0;
    mu = 1e-2;
    recurrentLayerUpdater::setMomentum();
}

void recurrentLayerUpdater::setMomentum(){
    if(iter <= 30){
        momentum_W = momentum_w_init;
        momentum_second_derivative = momentum_d2_init;
    }else{
        momentum_W = momentum_w_adjust;
        momentum_second_derivative = momentum_d2_adjust;
    }
}

void recurrentLayerUpdater::update(std::vector<Rl> &HiddenLayers, int iter_num){
    iter = iter_num;
    if(iter > 30) recurrentLayerUpdater::setMomentum();
    for(int i = 0; i < HiddenLayers.size(); ++i){
        second_derivative_W[i] = momentum_second_derivative * second_derivative_W[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Wd2;
        second_derivative_U[i] = momentum_second_derivative * second_derivative_U[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Ud2;
        learning_rate_W = HiddenLayers[i].lr_W / (second_derivative_W[i] + mu);
        learning_rate_U = HiddenLayers[i].lr_U / (second_derivative_U[i] + mu);
        velocity_W[i] = velocity_W[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Wgrad.mul(learning_rate_W);
        velocity_U[i] = velocity_U[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Ugrad.mul(learning_rate_U);
        HiddenLayers[i].W -= velocity_W[i];
        HiddenLayers[i].U -= velocity_U[i];
    }
}

// LSTM layer updater
LSTMLayerUpdater::LSTMLayerUpdater(std::vector<LSTMl> &HiddenLayers){
    init(HiddenLayers);
}

LSTMLayerUpdater::~LSTMLayerUpdater(){

    velocity_W_input.clear();
    std::vector<Mat>().swap(velocity_W_input);
    velocity_W_forget.clear();
    std::vector<Mat>().swap(velocity_W_forget);
    velocity_W_cell.clear();
    std::vector<Mat>().swap(velocity_W_cell);
    velocity_W_output.clear();
    std::vector<Mat>().swap(velocity_W_output);

    velocity_U_input.clear();
    std::vector<Mat>().swap(velocity_U_input);
    velocity_U_forget.clear();
    std::vector<Mat>().swap(velocity_U_forget);
    velocity_U_cell.clear();
    std::vector<Mat>().swap(velocity_U_cell);
    velocity_U_output.clear();
    std::vector<Mat>().swap(velocity_U_output);

    velocity_V_input.clear();
    std::vector<diagonalMatrix*>().swap(velocity_V_input);
    velocity_V_forget.clear();
    std::vector<diagonalMatrix*>().swap(velocity_V_forget);
    velocity_V_output.clear();
    std::vector<diagonalMatrix*>().swap(velocity_V_output);
    
    second_derivative_W_input.clear();
    std::vector<Mat>().swap(second_derivative_W_input);
    second_derivative_W_forget.clear();
    std::vector<Mat>().swap(second_derivative_W_forget);
    second_derivative_W_cell.clear();
    std::vector<Mat>().swap(second_derivative_W_cell);
    second_derivative_W_output.clear();
    std::vector<Mat>().swap(second_derivative_W_output);

    second_derivative_U_input.clear();
    std::vector<Mat>().swap(second_derivative_U_input);
    second_derivative_U_forget.clear();
    std::vector<Mat>().swap(second_derivative_U_forget);
    second_derivative_U_cell.clear();
    std::vector<Mat>().swap(second_derivative_U_cell);
    second_derivative_U_output.clear();
    std::vector<Mat>().swap(second_derivative_U_output);

    second_derivative_V_input.clear();
    std::vector<diagonalMatrix*>().swap(second_derivative_V_input);
    second_derivative_V_forget.clear();
    std::vector<diagonalMatrix*>().swap(second_derivative_V_forget);
    second_derivative_V_output.clear();
    std::vector<diagonalMatrix*>().swap(second_derivative_V_output);

    learning_rate_W.release();
    learning_rate_U.release();
}

void LSTMLayerUpdater::init(std::vector<LSTMl> &HiddenLayers){

    for(int i = 0; i < HiddenLayers.size(); ++i){
        Mat tmpW = Mat::zeros(HiddenLayers[i].W_input.size(), CV_64FC1);
        Mat tmpU = Mat::zeros(HiddenLayers[i].U_input.size(), CV_64FC1);
        diagonalMatrix *tmpdm;

        velocity_W_input.push_back(tmpW);
        velocity_W_forget.push_back(tmpW);
        velocity_W_cell.push_back(tmpW);
        velocity_W_output.push_back(tmpW);
        
        velocity_U_input.push_back(tmpU);
        velocity_U_forget.push_back(tmpU);
        velocity_U_cell.push_back(tmpU);
        velocity_U_output.push_back(tmpU);

        tmpdm = new diagonalMatrix(HiddenLayers[i].V_input -> diagonal.rows);
        velocity_V_input.push_back(tmpdm);
        tmpdm = new diagonalMatrix(HiddenLayers[i].V_input -> diagonal.rows);
        velocity_V_forget.push_back(tmpdm);
        tmpdm = new diagonalMatrix(HiddenLayers[i].V_input -> diagonal.rows);
        velocity_V_output.push_back(tmpdm);

        second_derivative_W_input.push_back(tmpW);
        second_derivative_W_forget.push_back(tmpW);
        second_derivative_W_cell.push_back(tmpW);
        second_derivative_W_output.push_back(tmpW);
        
        second_derivative_U_input.push_back(tmpU);
        second_derivative_U_forget.push_back(tmpU);
        second_derivative_U_cell.push_back(tmpU);
        second_derivative_U_output.push_back(tmpU);

        tmpdm = new diagonalMatrix(HiddenLayers[i].V_input -> diagonal.rows);
        second_derivative_V_input.push_back(tmpdm);
        tmpdm = new diagonalMatrix(HiddenLayers[i].V_input -> diagonal.rows);
        second_derivative_V_forget.push_back(tmpdm);
        tmpdm = new diagonalMatrix(HiddenLayers[i].V_input -> diagonal.rows);
        second_derivative_V_output.push_back(tmpdm);
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

        second_derivative_W_input[i] = momentum_second_derivative * second_derivative_W_input[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Wd2_input;
        second_derivative_W_forget[i] = momentum_second_derivative * second_derivative_W_forget[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Wd2_forget;
        second_derivative_W_cell[i] = momentum_second_derivative * second_derivative_W_cell[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Wd2_cell;
        second_derivative_W_output[i] = momentum_second_derivative * second_derivative_W_forget[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Wd2_output;
        
        second_derivative_U_input[i] = momentum_second_derivative * second_derivative_U_input[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Ud2_input;
        second_derivative_U_forget[i] = momentum_second_derivative * second_derivative_U_forget[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Ud2_forget;
        second_derivative_U_cell[i] = momentum_second_derivative * second_derivative_U_cell[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Ud2_cell;
        second_derivative_U_output[i] = momentum_second_derivative * second_derivative_U_forget[i] + (1.0 - momentum_second_derivative) * HiddenLayers[i].Ud2_output;
        
        second_derivative_V_input[i] -> full = momentum_second_derivative * second_derivative_V_input[i] -> full + (1.0 - momentum_second_derivative) * HiddenLayers[i].Vd2_input -> full;
        second_derivative_V_input[i] -> update(UPDATE_FROM_FULL);
        second_derivative_V_forget[i] -> full = momentum_second_derivative * second_derivative_V_forget[i] -> full + (1.0 - momentum_second_derivative) * HiddenLayers[i].Vd2_forget -> full;
        second_derivative_V_forget[i] -> update(UPDATE_FROM_FULL);
        second_derivative_V_output[i] -> full = momentum_second_derivative * second_derivative_V_output[i] -> full + (1.0 - momentum_second_derivative) * HiddenLayers[i].Vd2_output -> full;
        second_derivative_V_output[i] -> update(UPDATE_FROM_FULL);
        
        //*/
        learning_rate_W = HiddenLayers[i].lr_W / (second_derivative_W_input[i] + mu);
        learning_rate_U = HiddenLayers[i].lr_U / (second_derivative_U_input[i] + mu);
        learning_rate_V = HiddenLayers[i].lr_V / (second_derivative_V_input[i] -> full + mu);
        velocity_W_input[i] = velocity_W_input[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Wgrad_input.mul(learning_rate_W);
        velocity_U_input[i] = velocity_U_input[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Ugrad_input.mul(learning_rate_U);
        
        velocity_V_input[i] -> full = velocity_V_input[i] -> full * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Vgrad_input -> full.mul(learning_rate_V);
        velocity_V_input[i] -> update(UPDATE_FROM_FULL);
        //*/
       
        learning_rate_W = HiddenLayers[i].lr_W / (second_derivative_W_forget[i] + mu);
        learning_rate_U = HiddenLayers[i].lr_U / (second_derivative_U_forget[i] + mu);
        learning_rate_V = HiddenLayers[i].lr_V / (second_derivative_V_forget[i] -> full + mu);
        velocity_W_forget[i] = velocity_W_forget[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Wgrad_forget.mul(learning_rate_W);
        velocity_U_forget[i] = velocity_U_forget[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Ugrad_forget.mul(learning_rate_U);
        
        velocity_V_forget[i] -> full = velocity_V_forget[i] -> full * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Vgrad_forget -> full.mul(learning_rate_V);
        velocity_V_forget[i] -> update(UPDATE_FROM_FULL);
        //*/
        
        learning_rate_W = HiddenLayers[i].lr_W / (second_derivative_W_output[i] + mu);
        learning_rate_U = HiddenLayers[i].lr_U / (second_derivative_U_output[i] + mu);
        learning_rate_V = HiddenLayers[i].lr_V / (second_derivative_V_output[i] -> full + mu);
        velocity_W_output[i] = velocity_W_output[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Wgrad_output.mul(learning_rate_W);
        velocity_U_output[i] = velocity_U_output[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Ugrad_output.mul(learning_rate_U);
        
        velocity_V_output[i] -> full = velocity_V_output[i] -> full * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Vgrad_output -> full.mul(learning_rate_V);
        velocity_V_output[i] -> update(UPDATE_FROM_FULL);
        //*/
        
        learning_rate_W = HiddenLayers[i].lr_W / (second_derivative_W_cell[i] + mu);
        learning_rate_U = HiddenLayers[i].lr_U / (second_derivative_U_cell[i] + mu);
        velocity_W_cell[i] = velocity_W_cell[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Wgrad_cell.mul(learning_rate_W);
        velocity_U_cell[i] = velocity_U_cell[i] * momentum_W + (1.0 - momentum_W) * HiddenLayers[i].Ugrad_cell.mul(learning_rate_U);
        
        HiddenLayers[i].W_input -= velocity_W_input[i];
        HiddenLayers[i].W_forget -= velocity_W_forget[i];
        HiddenLayers[i].W_cell -= velocity_W_cell[i];
        HiddenLayers[i].W_output -= velocity_W_output[i];

        HiddenLayers[i].U_input -= velocity_U_input[i];
        HiddenLayers[i].U_forget -= velocity_U_forget[i];
        HiddenLayers[i].U_cell -= velocity_U_cell[i];
        HiddenLayers[i].U_output -= velocity_U_output[i];

        HiddenLayers[i].V_input -> full -= velocity_V_input[i] -> full;
        HiddenLayers[i].V_forget -> full -= velocity_V_forget[i] -> full;
        HiddenLayers[i].V_output -> full -= velocity_V_output[i] -> full;

        HiddenLayers[i].V_input -> update(UPDATE_FROM_FULL);
        HiddenLayers[i].V_forget -> update(UPDATE_FROM_FULL);
        HiddenLayers[i].V_output -> update(UPDATE_FROM_FULL);
        //*/
    }
}



