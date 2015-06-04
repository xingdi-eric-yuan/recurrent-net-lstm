#include "cost_gradient.h"

using namespace cv;
using namespace std;

// Long short-term memory node
bool 
getNetworkCost(std::vector<Mat> &x, Mat &y, std::vector<LSTMl> &hLayers, Smr &smr){

    int T = x.size();
    int nSamples = x[0].cols;
    // hidden layer forward
    std::vector<std::vector<Mat> > nonlin_input;
    std::vector<std::vector<Mat> > nonlin_forget;
    std::vector<std::vector<Mat> > nonlin_cell;
    std::vector<std::vector<Mat> > nonlin_output;

    std::vector<std::vector<Mat> > acti_input;
    std::vector<std::vector<Mat> > acti_forget;
    std::vector<std::vector<Mat> > acti_cell;
    std::vector<std::vector<Mat> > acti_output;
    std::vector<std::vector<Mat> > output_h;

    std::vector<Mat> tmp_vec(T);
    output_h.push_back(tmp_vec);
    for(int i = 0; i < T; ++i){
        x[i].copyTo(output_h[0][i]);
    }
    for(int i = 1; i <= hiddenConfig.size(); ++i){

        // for each hidden layer
        nonlin_input.push_back(tmp_vec);
        nonlin_forget.push_back(tmp_vec);
        nonlin_cell.push_back(tmp_vec);
        nonlin_output.push_back(tmp_vec);
        acti_input.push_back(tmp_vec);
        acti_forget.push_back(tmp_vec);
        acti_cell.push_back(tmp_vec);
        acti_output.push_back(tmp_vec);
        output_h.push_back(tmp_vec);

        for(int j = 0; j < T; ++j){
            // for each time slot
            Mat tmp_input, tmp_forget, tmp_cell, tmp_output;
            tmp_input = hLayers[i - 1].U_input * output_h[i - 1][j];
            tmp_forget = hLayers[i - 1].U_forget * output_h[i - 1][j];
            tmp_cell = hLayers[i - 1].U_cell * output_h[i - 1][j];
            tmp_output = hLayers[i - 1].U_output * output_h[i - 1][j];
            if(j > 0) {
                tmp_input += hLayers[i - 1].W_input * output_h[i][j - 1];
                tmp_input += hLayers[i - 1].V_input -> full * acti_cell[i - 1][j - 1];
                tmp_forget += hLayers[i - 1].W_forget * output_h[i][j - 1];
                tmp_forget += hLayers[i - 1].V_forget -> full * acti_cell[i - 1][j - 1];
                tmp_cell += hLayers[i - 1].W_cell * output_h[i][j - 1];
                tmp_output += hLayers[i - 1].W_output * output_h[i][j - 1];
            }
            tmp_input.copyTo(nonlin_input[i - 1][j]);
            tmp_forget.copyTo(nonlin_forget[i - 1][j]);
            tmp_cell.copyTo(nonlin_cell[i - 1][j]);
            tmp_input = nonLinearity(tmp_input);
            tmp_forget = nonLinearity(tmp_forget);
            tmp_cell = sigmoid(tmp_cell);
            tmp_input.copyTo(acti_input[i - 1][j]);
            tmp_forget.copyTo(acti_forget[i - 1][j]);
            tmp_cell = tmp_cell.mul(tmp_input);
            if(j > 0){
                tmp_cell += tmp_forget.mul(acti_cell[i - 1][j - 1]);
            }
            tmp_cell.copyTo(acti_cell[i - 1][j]);
            tmp_output += hLayers[i - 1].V_output -> full * tmp_cell;
            tmp_output.copyTo(nonlin_output[i - 1][j]);
            tmp_output = nonLinearity(tmp_output);
            tmp_output.copyTo(acti_output[i - 1][j]);
            tmp_output = tmp_output.mul(sigmoid(tmp_cell));
            tmp_output.copyTo(output_h[i][j]);
        }
    }
    // softmax layer forward
    std::vector<Mat> p(T);
    for(int i = 0; i < T; ++i){
        Mat M = smr.W * output_h[output_h.size() - 1][i];
        M -= repeat(reduce(M, 0, CV_REDUCE_MAX), M.rows, 1);
        M = exp(M);
        Mat tmpp = divide(M, repeat(reduce(M, 0, CV_REDUCE_SUM), M.rows, 1));
        tmpp.copyTo(p[i]);
    }
    std::vector<Mat> groundTruth(T);
    for(int i = 0; i < T; ++i){
        Mat tmpgroundTruth = Mat::zeros(softmaxConfig.NumClasses, nSamples, CV_64FC1);
        for(int j = 0; j < nSamples; j++){
            tmpgroundTruth.ATD(y.ATD(i, j), j) = 1.0;
        }
        tmpgroundTruth.copyTo(groundTruth[i]);
    }
    double J1 = 0.0;
    for(int i = 0; i < T; i++){
        J1 +=  - sum1(groundTruth[i].mul(log(p[i])));
    }
    J1 /= nSamples;
    if(prev_cost != -1.0 && J1 >= (prev_cost * 2.0)){
        // something's wrong
        cout<<endl;
        //cout.flags(ios::scientific);
        cout.precision(6);
        for(int i = 0; i < T; i++){
            cout<<"-----------------------------"<<endl;
            if(i > 0){
                cout<<"output_1_  "<<i<<": "<<endl<<" "<<hLayers[0].U_output * output_h[0][i]<<endl;
                cout<<"output_2_  "<<i<<": "<<endl<<" "<<hLayers[0].W_output * output_h[1][i - 1]<<endl;
                cout<<"output_3_  "<<i<<": "<<endl<<" "<<hLayers[0].V_output -> full * acti_cell[0][i]<<endl;
            }
            cout<<"output"<<i<<": "<<endl<<" "<<acti_output[acti_output.size() - 1][i]<<endl;
            cout<<"groundTruth"<<i<<": "<<endl<<" "<<groundTruth[i]<<endl;
            Mat M = smr.W * output_h[output_h.size() - 1][i];
            cout<<"M--"<<i<<": "<<endl<<" "<<M<<endl;
            M -= repeat(reduce(M, 0, CV_REDUCE_MAX), M.rows, 1);
            cout<<"Mred"<<i<<": "<<endl<<" "<<M<<endl;
            M = exp(M);
            cout<<"Mexp"<<i<<": "<<endl<<" "<<M<<endl;
            cout<<"p"<<i<<": "<<endl<<" "<<p[i]<<endl;
            cout<<"J"<<i<<": "<<endl<<" "<<groundTruth[i].mul(log(p[i]))<<endl;
            cout<<"output_h"<<i<<": "<<endl<<" "<<output_h[output_h.size() - 1][i]<<endl;
            cout<<"-----------------------------"<<endl;
        }
        //cout<<output_h[0][2]<<endl;
        //cout<<hLayers[0].V_output -> diagonal<<endl;
        return false;
        // exit(0);
    }else{
        prev_cost = J1;
    }
    //*/
    double J2 = sum1(pow(smr.W, 2.0)) * softmaxConfig.WeightDecay / 2;
    double J3 = 0.0; 
    double J4 = 0.0;
    double J5 = 0.0;
    //cout<<endl;
    for(int hl = 0; hl < hLayers.size(); hl++){
        //cout<<"W input : "<<sum1(pow(hLayers[hl].W_input, 2.0)) * hiddenConfig[hl].WeightDecay / 2<<endl;
        //cout<<"W forget : "<<sum1(pow(hLayers[hl].W_forget, 2.0)) * hiddenConfig[hl].WeightDecay / 2<<endl;
        //cout<<"W cell : "<<sum1(pow(hLayers[hl].W_cell, 2.0)) * hiddenConfig[hl].WeightDecay / 2<<endl;
        //cout<<"W output : "<<sum1(pow(hLayers[hl].W_output, 2.0)) * hiddenConfig[hl].WeightDecay / 2<<endl;
        J3 += sum1(pow(hLayers[hl].W_input, 2.0)) * hiddenConfig[hl].WeightDecay / 2;
        J3 += sum1(pow(hLayers[hl].W_forget, 2.0)) * hiddenConfig[hl].WeightDecay / 2;
        J3 += sum1(pow(hLayers[hl].W_cell, 2.0)) * hiddenConfig[hl].WeightDecay / 2;
        J3 += sum1(pow(hLayers[hl].W_output, 2.0)) * hiddenConfig[hl].WeightDecay / 2;
    }
    for(int hl = 0; hl < hLayers.size(); hl++){
        //cout<<"U input : "<<sum1(pow(hLayers[hl].U_input, 2.0)) * hiddenConfig[hl].WeightDecay / 2<<endl;
        //cout<<"U forget : "<<sum1(pow(hLayers[hl].U_forget, 2.0)) * hiddenConfig[hl].WeightDecay / 2<<endl;
        //cout<<"U cell : "<<sum1(pow(hLayers[hl].U_cell, 2.0)) * hiddenConfig[hl].WeightDecay / 2<<endl;
        //cout<<"U output : "<<sum1(pow(hLayers[hl].U_output, 2.0)) * hiddenConfig[hl].WeightDecay / 2<<endl;
        J4 += sum1(pow(hLayers[hl].U_input, 2.0)) * hiddenConfig[hl].WeightDecay / 2;
        J4 += sum1(pow(hLayers[hl].U_forget, 2.0)) * hiddenConfig[hl].WeightDecay / 2;
        J4 += sum1(pow(hLayers[hl].U_cell, 2.0)) * hiddenConfig[hl].WeightDecay / 2;
        J4 += sum1(pow(hLayers[hl].U_output, 2.0)) * hiddenConfig[hl].WeightDecay / 2;
    }
    
    for(int hl = 0; hl < hLayers.size(); hl++){
        J5 += sum1(pow(hLayers[hl].V_input -> diagonal, 2.0)) * hiddenConfig[hl].WeightDecay / 2;
        J5 += sum1(pow(hLayers[hl].V_forget -> diagonal, 2.0)) * hiddenConfig[hl].WeightDecay / 2;
        J5 += sum1(pow(hLayers[hl].V_output -> diagonal, 2.0)) * hiddenConfig[hl].WeightDecay / 2;
    }//*/
    smr.cost = J1 + J2 + J3 + J4 + J5;
    if(!is_gradient_checking) 
        cout<<", J1 = "<<J1<<", J2 = "<<J2<<", J3 = "<<J3<<", J4 = "<<J4<<", J5 = "<<J5<<", Cost = "<<smr.cost<<endl;

    // softmax layer backward
    Mat tmp, tmp2;
    tmp = - (groundTruth[0] - p[0]) * output_h[output_h.size() - 1][0].t();
    tmp2 = pow((groundTruth[0] - p[0]), 2.0) * pow(output_h[output_h.size() - 1][0].t(), 2.0);
    for(int i = 1; i < T; ++i){
        tmp += - (groundTruth[i] - p[i]) * output_h[output_h.size() - 1][i].t();
        tmp2 += pow((groundTruth[i] - p[i]), 2.0) * pow(output_h[output_h.size() - 1][i].t(), 2.0);
    }
    smr.Wgrad =  tmp / nSamples + softmaxConfig.WeightDecay * smr.W;
    smr.Wd2 = tmp2 / nSamples + softmaxConfig.WeightDecay;

    // hidden layer backward
    std::vector<std::vector<Mat> > delta_input;
    std::vector<std::vector<Mat> > deltad2_input;
    std::vector<std::vector<Mat> > delta_forget;
    std::vector<std::vector<Mat> > deltad2_forget;
    std::vector<std::vector<Mat> > delta_cell;
    std::vector<std::vector<Mat> > deltad2_cell;
    std::vector<std::vector<Mat> > delta_output;
    std::vector<std::vector<Mat> > deltad2_output;
    std::vector<std::vector<Mat> > epsilon_output;
    std::vector<std::vector<Mat> > epsilond2_output;
    std::vector<std::vector<Mat> > epsilon_state;
    std::vector<std::vector<Mat> > epsilond2_state;
    for(int i = 0; i < output_h.size(); i++){
        delta_input.push_back(tmp_vec);
        deltad2_input.push_back(tmp_vec);
        delta_forget.push_back(tmp_vec);
        deltad2_forget.push_back(tmp_vec);
        delta_cell.push_back(tmp_vec);
        deltad2_cell.push_back(tmp_vec);
        delta_output.push_back(tmp_vec);
        deltad2_output.push_back(tmp_vec);
        epsilon_output.push_back(tmp_vec);
        epsilond2_output.push_back(tmp_vec);
        epsilon_state.push_back(tmp_vec);
        epsilond2_state.push_back(tmp_vec);
    }
    for(int i = T - 1; i >= 0; --i){
        // cell output
        tmp = -smr.W.t() * (groundTruth[i] - p[i]);
        tmp2 = pow(smr.W.t(), 2.0) * pow((groundTruth[i] - p[i]), 2.0);
        if(i < T - 1){            
            tmp += hLayers[hLayers.size() - 1].W_cell.t() * delta_cell[delta_cell.size() - 1][i + 1];
            tmp += hLayers[hLayers.size() - 1].W_input.t() * delta_input[delta_input.size() - 1][i + 1];
            tmp += hLayers[hLayers.size() - 1].W_forget.t() * delta_forget[delta_forget.size() - 1][i + 1];
            tmp += hLayers[hLayers.size() - 1].W_output.t() * delta_output[delta_output.size() - 1][i + 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].W_cell.t(), 2.0) * deltad2_cell[deltad2_cell.size() - 1][i + 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].W_input.t(), 2.0) * deltad2_input[deltad2_input.size() - 1][i + 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].W_forget.t(), 2.0) * deltad2_forget[deltad2_forget.size() - 1][i + 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].W_output.t(), 2.0) * deltad2_output[deltad2_output.size() - 1][i + 1];
        }
        tmp.copyTo(epsilon_output[epsilon_output.size() - 1][i]);
        tmp2.copyTo(epsilond2_output[epsilond2_output.size() - 1][i]);
        // output gates
        tmp = tmp.mul(sigmoid(acti_cell[acti_cell.size() - 1][i]));
        tmp = tmp.mul(dnonLinearity(nonlin_output[nonlin_output.size() - 1][i]));
        tmp2 = tmp2.mul(pow(sigmoid(acti_cell[acti_cell.size() - 1][i]), 2.0));
        tmp2 = tmp2.mul(pow(dnonLinearity(nonlin_output[nonlin_output.size() - 1][i]), 2.0));
        tmp.copyTo(delta_output[delta_output.size() - 1][i]);
        tmp2.copyTo(deltad2_output[deltad2_output.size() - 1][i]);
        // states
        tmp = acti_output[acti_output.size() - 1][i].mul(dsigmoid(acti_cell[acti_cell.size() - 1][i]));
        tmp = tmp.mul(epsilon_output[epsilon_output.size() - 1][i]);
        tmp2 = pow(acti_output[acti_output.size() - 1][i], 2.0).mul(pow(dsigmoid(acti_cell[acti_cell.size() - 1][i]), 2.0));
        tmp2 = tmp2.mul(epsilond2_output[epsilond2_output.size() - 1][i]);
        if(i < T - 1){
            tmp += acti_forget[acti_forget.size() - 1][i + 1].mul(epsilon_state[epsilon_state.size() - 1][i + 1]);
            tmp += hLayers[hLayers.size() - 1].V_input -> full.t() * delta_input[delta_input.size() - 1][i + 1];
            tmp += hLayers[hLayers.size() - 1].V_forget -> full.t() * delta_forget[delta_forget.size() - 1][i + 1];
            tmp2 += pow(acti_forget[acti_forget.size() - 1][i + 1], 2.0).mul(epsilond2_state[epsilond2_state.size() - 1][i + 1]);
            tmp2 += pow(hLayers[hLayers.size() - 1].V_input -> full.t(), 2.0) * deltad2_input[deltad2_input.size() - 1][i + 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].V_forget -> full.t(), 2.0) * deltad2_forget[deltad2_forget.size() - 1][i + 1];
        }
        tmp += hLayers[hLayers.size() - 1].V_output -> full.t() * delta_output[delta_output.size() - 1][i];
        tmp2 += pow(hLayers[hLayers.size() - 1].V_output -> full.t(), 2.0) * deltad2_output[deltad2_output.size() - 1][i];
        tmp.copyTo(epsilon_state[epsilon_state.size() - 1][i]);
        tmp2.copyTo(epsilond2_state[epsilond2_state.size() - 1][i]);
        // cells
        tmp = acti_input[acti_input.size() - 1][i].mul(dsigmoid(nonlin_cell[nonlin_cell.size() - 1][i]));
        tmp = tmp.mul(epsilon_state[epsilon_state.size() - 1][i]);
        tmp2 = pow(acti_input[acti_input.size() - 1][i], 2.0).mul(pow(dsigmoid(nonlin_cell[nonlin_cell.size() - 1][i]), 2.0));
        tmp2 = tmp2.mul(epsilond2_state[epsilond2_state.size() - 1][i]);
        tmp.copyTo(delta_cell[delta_cell.size() - 1][i]);
        tmp2.copyTo(deltad2_cell[deltad2_cell.size() - 1][i]);
        // forget gates
        if(i > 0){
            tmp = acti_cell[acti_cell.size() - 1][i - 1].mul(epsilon_state[epsilon_state.size() - 1][i]);
            tmp = tmp.mul(dnonLinearity(nonlin_forget[nonlin_forget.size() - 1][i]));
            tmp2 = pow(acti_cell[acti_cell.size() - 1][i - 1], 2.0).mul(epsilond2_state[epsilond2_state.size() - 1][i]);
            tmp2 = tmp2.mul(pow(dnonLinearity(nonlin_forget[nonlin_forget.size() - 1][i]), 2.0));
        }else{
            tmp = Mat::zeros(nonlin_forget[nonlin_forget.size() - 1][i].size(), CV_64FC1);
            tmp2 = Mat::zeros(nonlin_forget[nonlin_forget.size() - 1][i].size(), CV_64FC1);
        }
        tmp.copyTo(delta_forget[delta_forget.size() - 1][i]);
        tmp2.copyTo(deltad2_forget[deltad2_forget.size() - 1][i]);
        // input gates
        tmp = epsilon_state[epsilon_state.size() - 1][i].mul(sigmoid(nonlin_cell[nonlin_cell.size() - 1][i]));
        tmp = tmp.mul(dnonLinearity(nonlin_input[nonlin_input.size() - 1][i]));
        tmp2 = epsilond2_state[epsilond2_state.size() - 1][i].mul(pow(sigmoid(nonlin_cell[nonlin_cell.size() - 1][i]), 2.0));
        tmp2 = tmp2.mul(pow(dnonLinearity(nonlin_input[nonlin_input.size() - 1][i]), 2.0));
        tmp.copyTo(delta_input[delta_input.size() - 1][i]);
        tmp2.copyTo(deltad2_input[deltad2_input.size() - 1][i]);
    }
    
    for(int i = delta_input.size() - 2; i > 0; --i){
        for(int j = T - 1; j >= 0; --j){
            // cell output
            tmp = hLayers[i].U_cell.t() * delta_cell[i + 1][j];
            tmp += hLayers[i].U_input.t() * delta_input[i + 1][j];
            tmp += hLayers[i].U_forget.t() * delta_forget[i + 1][j];
            tmp += hLayers[i].U_output.t() * delta_output[i + 1][j];
            tmp2 = pow(hLayers[i].U_cell.t(), 2.0) * deltad2_cell[i + 1][j];
            tmp2 += pow(hLayers[i].U_input.t(), 2.0) * deltad2_input[i + 1][j];
            tmp2 += pow(hLayers[i].U_forget.t(), 2.0) * deltad2_forget[i + 1][j];
            tmp2 += pow(hLayers[i].U_output.t(), 2.0) * deltad2_output[i + 1][j];
            if(j < T - 1){
                tmp += hLayers[i - 1].W_cell.t() * delta_cell[i][j + 1];
                tmp += hLayers[i - 1].W_input.t() * delta_input[i][j + 1];
                tmp += hLayers[i - 1].W_forget.t() * delta_forget[i][j + 1];
                tmp += hLayers[i - 1].W_output.t() * delta_output[i][j + 1];
                tmp2 += pow(hLayers[i - 1].W_cell.t(), 2.0) * deltad2_cell[i][j + 1];
                tmp2 += pow(hLayers[i - 1].W_input.t(), 2.0) * deltad2_input[i][j + 1];
                tmp2 += pow(hLayers[i - 1].W_forget.t(), 2.0) * deltad2_forget[i][j + 1];
                tmp2 += pow(hLayers[i - 1].W_output.t(), 2.0) * deltad2_output[i][j + 1];
            }
            tmp.copyTo(epsilon_output[i][j]);
            tmp2.copyTo(epsilond2_output[i][j]);
            // output gates
            tmp = tmp.mul(sigmoid(acti_cell[i - 1][j]));
            tmp = tmp.mul(dnonLinearity(nonlin_output[i - 1][j]));
            tmp2 = tmp2.mul(pow(sigmoid(acti_cell[i - 1][j]), 2.0));
            tmp2 = tmp2.mul(pow(dnonLinearity(nonlin_output[i - 1][j]), 2.0));
            tmp.copyTo(delta_output[i][j]);
            tmp2.copyTo(deltad2_output[i][j]);
            // states
            tmp = acti_output[i - 1][j].mul(dsigmoid(acti_cell[i - 1][j]));
            tmp = tmp.mul(epsilon_output[i][j]);
            tmp2 = pow(acti_output[i - 1][j], 2.0).mul(pow(dsigmoid(acti_cell[i - 1][j]), 2.0));
            tmp2 = tmp2.mul(epsilond2_output[i][j]);
            if(j < T - 1){
                tmp += acti_forget[i - 1][j + 1].mul(epsilon_state[i][j + 1]);
                tmp += hLayers[i - 1].V_input -> full.t() * delta_input[i][j + 1];
                tmp += hLayers[i - 1].V_forget -> full.t() * delta_forget[i][j + 1];
                tmp2 += pow(acti_forget[i - 1][j + 1], 2.0).mul(epsilond2_state[i][j + 1]);
                tmp2 += pow(hLayers[i - 1].V_input -> full.t(), 2.0) * deltad2_input[i][j + 1];
                tmp2 += pow(hLayers[i - 1].V_forget -> full.t(), 2.0) * deltad2_forget[i][j + 1];
            }
            tmp += hLayers[i - 1].V_output -> full.t() * delta_output[i][j];
            tmp2 += pow(hLayers[i - 1].V_output -> full.t(), 2.0) * deltad2_output[i][j];
            tmp.copyTo(epsilon_state[i][j]);
            tmp2.copyTo(epsilond2_state[i][j]);
            // cells
            tmp = acti_input[i - 1][j].mul(dsigmoid(nonlin_cell[i - 1][j]));
            tmp = tmp.mul(epsilon_state[i][j]);
            tmp2 = pow(acti_input[i - 1][j], 2.0).mul(pow(dsigmoid(nonlin_cell[i - 1][j]), 2.0));
            tmp2 = tmp2.mul(epsilond2_state[i][j]);
            tmp.copyTo(delta_cell[i][j]);
            tmp2.copyTo(deltad2_cell[i][j]);
            // forget gates
            if(j > 0){
                tmp = acti_cell[i - 1][j].mul(epsilon_state[i][j]);
                tmp = tmp.mul(dnonLinearity(nonlin_forget[i - 1][j]));
                tmp2 = pow(acti_cell[i - 1][j], 2.0).mul(epsilond2_state[i][j]);
                tmp2 = tmp2.mul(pow(dnonLinearity(nonlin_forget[i - 1][j]), 2.0));
            }else{
                tmp = Mat::zeros(nonlin_forget[i - 1][j].size(), CV_64FC1);
                tmp2 = Mat::zeros(nonlin_forget[i - 1][j].size(), CV_64FC1);
            }
            tmp.copyTo(delta_forget[i][j]);
            tmp2.copyTo(deltad2_forget[i][j]);
            // input gates
            tmp = epsilon_state[i][j].mul(sigmoid(nonlin_cell[i - 1][j]));
            tmp = tmp.mul(dnonLinearity(nonlin_input[i - 1][j]));
            tmp2 = epsilond2_state[i][j].mul(pow(sigmoid(nonlin_cell[i - 1][j]), 2.0));
            tmp2 = tmp2.mul(pow(dnonLinearity(nonlin_input[i - 1][j]), 2.0));
            tmp.copyTo(delta_input[i][j]);
            tmp2.copyTo(deltad2_input[i][j]);
        }
    }
//*/
    for(int i = hiddenConfig.size() - 1; i >= 0; i--){
        // U
        tmp = delta_input[i + 1][0] * output_h[i][0].t();
        tmp2 = deltad2_input[i + 1][0] * pow(output_h[i][0].t(), 2.0);
        for(int j = 1; j < T; ++j){
            tmp += delta_input[i + 1][j] * output_h[i][j].t();
            tmp2 += deltad2_input[i + 1][j] * pow(output_h[i][j].t(), 2.0);
        }
        hLayers[i].Ugrad_input = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].U_input;
        hLayers[i].Ud2_input = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_forget[i + 1][0] * output_h[i][0].t();
        tmp2 = deltad2_forget[i + 1][0] * pow(output_h[i][0].t(), 2.0);
        for(int j = 1; j < T; ++j){
            tmp += delta_forget[i + 1][j] * output_h[i][j].t();
            tmp2 += deltad2_forget[i + 1][j] * pow(output_h[i][j].t(), 2.0);
        }
        hLayers[i].Ugrad_forget = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].U_forget;
        hLayers[i].Ud2_forget = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_cell[i + 1][0] * output_h[i][0].t();
        tmp2 = deltad2_cell[i + 1][0] * pow(output_h[i][0].t(), 2.0);
        for(int j = 1; j < T; ++j){
            tmp += delta_cell[i + 1][j] * output_h[i][j].t();
            tmp2 += deltad2_cell[i + 1][j] * pow(output_h[i][j].t(), 2.0);
        }
        hLayers[i].Ugrad_cell = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].U_cell;
        hLayers[i].Ud2_cell = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_output[i + 1][0] * output_h[i][0].t();
        tmp2 = deltad2_output[i + 1][0] * pow(output_h[i][0].t(), 2.0);
        for(int j = 1; j < T; ++j){
            tmp += delta_output[i + 1][j] * output_h[i][j].t();
            tmp2 += deltad2_output[i + 1][j] * pow(output_h[i][j].t(), 2.0);
        }
        hLayers[i].Ugrad_output = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].U_output;
        hLayers[i].Ud2_output = tmp2 / nSamples + hiddenConfig[i].WeightDecay;
        // W
        tmp = delta_input[i + 1][T - 1] * output_h[i + 1][T - 2].t();
        tmp2 = deltad2_input[i + 1][T - 1] * pow(output_h[i + 1][T - 2].t(), 2.0);

        for(int j = T - 2; j > 0; j--){
            tmp += delta_input[i + 1][j] * output_h[i + 1][j - 1].t();
            tmp2 += deltad2_input[i + 1][j] * pow(output_h[i + 1][j - 1].t(), 2.0);
        }
        hLayers[i].Wgrad_input = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].W_input;
        hLayers[i].Wd2_input = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_forget[i + 1][T - 1] * output_h[i + 1][T - 2].t();
        tmp2 = deltad2_forget[i + 1][T - 1] * pow(output_h[i + 1][T - 2].t(), 2.0);
        for(int j = T - 2; j > 0; j--){
            tmp += delta_forget[i + 1][j] * output_h[i + 1][j - 1].t();
            tmp2 += deltad2_forget[i + 1][j] * pow(output_h[i + 1][j - 1].t(), 2.0);
        }
        hLayers[i].Wgrad_forget = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].W_forget;
        hLayers[i].Wd2_forget = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_cell[i + 1][T - 1] * output_h[i + 1][T - 2].t();
        tmp2 = deltad2_cell[i + 1][T - 1] * pow(output_h[i + 1][T - 2].t(), 2.0);
        for(int j = T - 2; j > 0; j--){
            tmp += delta_cell[i + 1][j] * output_h[i + 1][j - 1].t();
            tmp2 += deltad2_cell[i + 1][j] * pow(output_h[i + 1][j - 1].t(), 2.0);
        }
        hLayers[i].Wgrad_cell = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].W_cell;
        hLayers[i].Wd2_cell = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_output[i + 1][T - 1] * output_h[i + 1][T - 2].t();
        tmp2 = deltad2_output[i + 1][T - 1] * pow(output_h[i + 1][T - 2].t(), 2.0);
        for(int j = T - 2; j > 0; j--){
            tmp += delta_output[i + 1][j] * output_h[i + 1][j - 1].t();
            tmp2 += deltad2_output[i + 1][j] * pow(output_h[i + 1][j - 1].t(), 2.0);
        }
        hLayers[i].Wgrad_output = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].W_output;
        hLayers[i].Wd2_output = tmp2 / nSamples + hiddenConfig[i].WeightDecay;
        // V
        tmp = delta_input[i + 1][T - 1] * acti_cell[i][T - 2].t();
        tmp2 = deltad2_input[i + 1][T - 1] * pow(acti_cell[i][T - 2].t(), 2.0);
        for(int j = T - 2; j > 0; j--){
            tmp += delta_input[i + 1][j] * acti_cell[i][j - 1].t();
            tmp2 += deltad2_input[i + 1][j] * pow(acti_cell[i][j - 1].t(), 2.0);
        }
        hLayers[i].Vgrad_input -> full = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].V_input -> full;
        hLayers[i].Vgrad_input -> update(UPDATE_FROM_FULL);
        hLayers[i].Vgrad_input -> update(UPDATE_FROM_DIAG);
        hLayers[i].Vd2_input -> full = tmp2 / nSamples + hiddenConfig[i].WeightDecay;
        hLayers[i].Vd2_input -> update(UPDATE_FROM_FULL);
        hLayers[i].Vd2_input -> update(UPDATE_FROM_DIAG);

        tmp = delta_forget[i + 1][T - 1] * acti_cell[i][T - 2].t();
        tmp2 = deltad2_forget[i + 1][T - 1] * pow(acti_cell[i][T - 2].t(), 2.0);
        for(int j = T - 2; j > 0; j--){
            tmp += delta_forget[i + 1][j] * acti_cell[i][j - 1].t();
            tmp2 += deltad2_forget[i + 1][j] * pow(acti_cell[i][j - 1].t(), 2.0);
        }
        hLayers[i].Vgrad_forget -> full = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].V_forget -> full;
        hLayers[i].Vgrad_forget -> update(UPDATE_FROM_FULL);
        hLayers[i].Vgrad_forget -> update(UPDATE_FROM_DIAG);
        hLayers[i].Vd2_forget -> full = tmp2 / nSamples + hiddenConfig[i].WeightDecay;
        hLayers[i].Vd2_forget -> update(UPDATE_FROM_FULL);
        hLayers[i].Vd2_forget -> update(UPDATE_FROM_DIAG);

        tmp = delta_output[i + 1][T - 1] * acti_cell[i][T - 2].t();
        tmp2 = deltad2_output[i + 1][T - 1] * pow(acti_cell[i][T - 2].t(), 2.0);
        for(int j = T - 2; j > 0; j--){
            tmp += delta_output[i + 1][j] * acti_cell[i][j - 1].t();
            tmp2 += deltad2_output[i + 1][j] * pow(acti_cell[i][j - 1].t(), 2.0);
        }
        hLayers[i].Vgrad_output -> full = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].V_output -> full;
        hLayers[i].Vgrad_output -> update(UPDATE_FROM_FULL);
        hLayers[i].Vgrad_output -> update(UPDATE_FROM_DIAG);
        hLayers[i].Vd2_output -> full = tmp2 / nSamples + hiddenConfig[i].WeightDecay;
        hLayers[i].Vd2_output -> update(UPDATE_FROM_FULL);
        hLayers[i].Vd2_output -> update(UPDATE_FROM_DIAG);
        //*/
    }

    // destructor
    nonlin_input.clear();
    std::vector<std::vector<Mat> > ().swap(nonlin_input);
    nonlin_forget.clear();
    std::vector<std::vector<Mat> > ().swap(nonlin_forget);
    nonlin_cell.clear();
    std::vector<std::vector<Mat> > ().swap(nonlin_cell);
    nonlin_output.clear();
    std::vector<std::vector<Mat> > ().swap(nonlin_output);
    acti_input.clear();
    std::vector<std::vector<Mat> > ().swap(acti_input);
    acti_forget.clear();
    std::vector<std::vector<Mat> > ().swap(acti_forget);
    acti_cell.clear();
    std::vector<std::vector<Mat> > ().swap(acti_cell);
    acti_output.clear();
    std::vector<std::vector<Mat> > ().swap(acti_output);
    output_h.clear();
    std::vector<std::vector<Mat> > ().swap(output_h);

    delta_input.clear();
    std::vector<std::vector<Mat> > ().swap(delta_input);
    deltad2_input.clear();
    std::vector<std::vector<Mat> > ().swap(deltad2_input);
    delta_forget.clear();
    std::vector<std::vector<Mat> > ().swap(delta_forget);
    deltad2_forget.clear();
    std::vector<std::vector<Mat> > ().swap(deltad2_forget);
    delta_cell.clear();
    std::vector<std::vector<Mat> > ().swap(delta_cell);
    deltad2_cell.clear();
    std::vector<std::vector<Mat> > ().swap(deltad2_cell);
    delta_output.clear();
    std::vector<std::vector<Mat> > ().swap(delta_output);
    deltad2_output.clear();
    std::vector<std::vector<Mat> > ().swap(deltad2_output);
    epsilon_output.clear();
    std::vector<std::vector<Mat> > ().swap(epsilon_output);
    epsilond2_output.clear();
    std::vector<std::vector<Mat> > ().swap(epsilond2_output);
    epsilon_state.clear();
    std::vector<std::vector<Mat> > ().swap(epsilon_state);
    epsilond2_state.clear();
    std::vector<std::vector<Mat> > ().swap(epsilond2_state);

    groundTruth.clear();
    std::vector<Mat>().swap(groundTruth);
    p.clear();
    std::vector<Mat>().swap(p);
    return true;

}

// traditional uni-directional recurrent node
void 
getNetworkCost(std::vector<Mat> &x, Mat &y, std::vector<Rl> &hLayers, Smr &smr){

    int T = x.size();
    int nSamples = x[0].cols;
    // hidden layer forward
    std::vector<std::vector<Mat> > nonlin;
    std::vector<std::vector<Mat> > acti;
    std::vector<std::vector<Mat> > bernoulli;

    std::vector<Mat> tmp_vec;
    acti.push_back(tmp_vec);
    for(int i = 0; i < T; ++i){
        acti[0].push_back(x[i]);
    }
    for(int i = 1; i <= hiddenConfig.size(); ++i){
        // for each hidden layer
        acti.push_back(tmp_vec);
        nonlin.push_back(tmp_vec);
        bernoulli.push_back(tmp_vec);
        for(int j = 0; j < T; ++j){
            // for each time slot
            Mat tmpacti = hLayers[i - 1].U * acti[i - 1][j];
            if(j > 0) tmpacti += hLayers[i - 1].W * acti[i][j - 1];
            nonlin[i - 1].push_back(tmpacti);
            tmpacti = ReLU(tmpacti);
            acti[i].push_back(tmpacti);
        }
    }
    // softmax layer forward
    std::vector<Mat> p;
    for(int i = 0; i < T; ++i){
        Mat M = smr.W * acti[acti.size() - 1][i];
        M -= repeat(reduce(M, 0, CV_REDUCE_MAX), M.rows, 1);
        M = exp(M);
        Mat tmpp = divide(M, repeat(reduce(M, 0, CV_REDUCE_SUM), M.rows, 1));
        p.push_back(tmpp);
    }
    std::vector<Mat> groundTruth;
    for(int i = 0; i < T; ++i){
        Mat tmpgroundTruth = Mat::zeros(softmaxConfig.NumClasses, nSamples, CV_64FC1);
        for(int j = 0; j < nSamples; j++){
            tmpgroundTruth.ATD(y.ATD(i, j), j) = 1.0;
        }
        groundTruth.push_back(tmpgroundTruth);
    }
    double J1 = 0.0;
    for(int i = 0; i < T; i++){
        J1 +=  - sum1(groundTruth[i].mul(log(p[i])));
    }
    J1 /= nSamples;
    double J2 = sum1(pow(smr.W, 2.0)) * softmaxConfig.WeightDecay / 2;
    double J3 = 0.0; 
    double J4 = 0.0;
    for(int hl = 0; hl < hLayers.size(); hl++){
        J3 += sum1(pow(hLayers[hl].W, 2.0)) * hiddenConfig[hl].WeightDecay / 2;
    }
    for(int hl = 0; hl < hLayers.size(); hl++){
        J4 += sum1(pow(hLayers[hl].U, 2.0)) * hiddenConfig[hl].WeightDecay / 2;
    }
    smr.cost = J1 + J2 + J3 + J4;
    if(!is_gradient_checking) 
        cout<<", J1 = "<<J1<<", J2 = "<<J2<<", J3 = "<<J3<<", J4 = "<<J4<<", Cost = "<<smr.cost<<endl;

    // softmax layer backward
    Mat tmp, tmp2;
    tmp = - (groundTruth[0] - p[0]) * acti[acti.size() - 1][0].t();
    for(int i = 1; i < T; ++i){
        tmp += - (groundTruth[i] - p[i]) * acti[acti.size() - 1][i].t();
    }
    smr.Wgrad =  tmp / nSamples + softmaxConfig.WeightDecay * smr.W;
    tmp = pow((groundTruth[0] - p[0]), 2.0) * pow(acti[acti.size() - 1][0].t(), 2.0);
    for(int i = 1; i < T; ++i){
        tmp += pow((groundTruth[i] - p[i]), 2.0) * pow(acti[acti.size() - 1][i].t(), 2.0);
    }
    smr.Wd2 = tmp / nSamples + softmaxConfig.WeightDecay;

    // hidden layer backward
    std::vector<std::vector<Mat> > delta(acti.size());
    std::vector<std::vector<Mat> > deltad2(acti.size());
    for(int i = 0; i < delta.size(); i++){
        delta[i].clear();
        deltad2[i].clear();
        Mat tmpmat;
        for(int j = 0; j < T; j++){
            delta[i].push_back(tmpmat);
            deltad2[i].push_back(tmpmat);
        }
    }

    for(int i = T - 1; i >= 0; --i){
        tmp = -smr.W.t() * (groundTruth[i] - p[i]);
        tmp2 = pow(smr.W.t(), 2.0) * pow((groundTruth[i] - p[i]), 2.0);
        if(i < T - 1){
            tmp += hLayers[hLayers.size() - 1].W.t() * delta[delta.size() - 1][i + 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].W.t(), 2.0) * deltad2[deltad2.size() - 1][i + 1];
        }
        tmp.copyTo(delta[delta.size() - 1][i]);
        tmp2.copyTo(deltad2[deltad2.size() - 1][i]);
        delta[delta.size() - 1][i] = delta[delta.size() - 1][i].mul(dReLU(nonlin[nonlin.size() - 1][i]));
        deltad2[deltad2.size() - 1][i] = deltad2[deltad2.size() - 1][i].mul(pow(dReLU(nonlin[nonlin.size() - 1][i]), 2.0));
    }

    for(int i = delta.size() - 2; i > 0; --i){
        for(int j = T - 1; j >= 0; --j){
            tmp = hLayers[i].U.t() * delta[i + 1][j];
            tmp2 = pow(hLayers[i].U.t(), 2.0) * deltad2[i + 1][j];
            if(j < T - 1){
                tmp += hLayers[i - 1].W.t() * delta[i][j + 1];
                tmp2 += pow(hLayers[i - 1].W.t(), 2.0) * deltad2[i][j + 1];
            }
            tmp.copyTo(delta[i][j]);
            tmp2.copyTo(deltad2[i][j]);
            delta[i][j] = delta[i][j].mul(dReLU(nonlin[i - 1][j]));
            deltad2[i][j] = deltad2[i][j].mul(pow(dReLU(nonlin[i - 1][j]), 2.0));
        }
    }
    for(int i = hiddenConfig.size() - 1; i >= 0; i--){
        tmp = delta[i + 1][0] * acti[i][0].t();
        tmp2 = deltad2[i + 1][0] * pow(acti[i][0].t(), 2.0);
        for(int j = 1; j < T; ++j){
            tmp += delta[i + 1][j] * acti[i][j].t();
            tmp2 += deltad2[i + 1][j] * pow(acti[i][j].t(), 2.0);
        }
        hLayers[i].Ugrad = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].U;
        hLayers[i].Ud2 = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta[i + 1][T - 1] * acti[i + 1][T - 2].t();
        tmp2 = deltad2[i + 1][T - 1] * pow(acti[i + 1][T - 2].t(), 2.0);
        for(int j = T - 2; j > 0; j--){
            tmp += delta[i + 1][j] * acti[i + 1][j - 1].t();
            tmp2 += deltad2[i + 1][j] * pow(acti[i + 1][j - 1].t(), 2.0);
        }
        hLayers[i].Wgrad = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].W;
        hLayers[i].Wd2 = tmp2 / nSamples + hiddenConfig[i].WeightDecay;
    }
    // destructor
    acti.clear();
    std::vector<std::vector<Mat> >().swap(acti);
    nonlin.clear();
    std::vector<std::vector<Mat> >().swap(nonlin);
    delta.clear();
    std::vector<std::vector<Mat> >().swap(delta);
    deltad2.clear();
    std::vector<std::vector<Mat> >().swap(deltad2);
    bernoulli.clear();
    std::vector<std::vector<Mat> >().swap(bernoulli);
    groundTruth.clear();
    std::vector<Mat>().swap(groundTruth);
    p.clear();
    std::vector<Mat>().swap(p);

}

