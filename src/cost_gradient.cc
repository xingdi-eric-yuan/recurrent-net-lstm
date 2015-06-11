#include "cost_gradient.h"

using namespace cv;
using namespace std;

// Long short-term memory node
bool 
getNetworkCost(std::vector<Mat> &x, Mat &y, std::vector<LSTMl> &hLayers, Smr &smr){

    int T = x.size();
    int nSamples = x[0].cols;
    // hidden layer forward
    std::vector<std::vector<Mat> > nonlin_input_left;
    std::vector<std::vector<Mat> > nonlin_forget_left;
    std::vector<std::vector<Mat> > nonlin_cell_left;
    std::vector<std::vector<Mat> > nonlin_output_left;

    std::vector<std::vector<Mat> > acti_input_left;
    std::vector<std::vector<Mat> > acti_forget_left;
    std::vector<std::vector<Mat> > acti_cell_left;
    std::vector<std::vector<Mat> > acti_output_left;
    std::vector<std::vector<Mat> > output_h_left;

    std::vector<std::vector<Mat> > nonlin_input_right;
    std::vector<std::vector<Mat> > nonlin_forget_right;
    std::vector<std::vector<Mat> > nonlin_cell_right;
    std::vector<std::vector<Mat> > nonlin_output_right;

    std::vector<std::vector<Mat> > acti_input_right;
    std::vector<std::vector<Mat> > acti_forget_right;
    std::vector<std::vector<Mat> > acti_cell_right;
    std::vector<std::vector<Mat> > acti_output_right;
    std::vector<std::vector<Mat> > output_h_right;

    std::vector<Mat> tmp_vec(T);
    output_h_left.push_back(tmp_vec);
    output_h_right.push_back(tmp_vec);
    for(int i = 0; i < T; ++i){
        x[i].copyTo(output_h_left[0][i]);
        x[i].copyTo(output_h_right[0][i]);
    }
    for(int i = 1; i <= hiddenConfig.size(); ++i){
        // for each hidden layer
        nonlin_input_left.push_back(tmp_vec);
        nonlin_forget_left.push_back(tmp_vec);
        nonlin_cell_left.push_back(tmp_vec);
        nonlin_output_left.push_back(tmp_vec);
        acti_input_left.push_back(tmp_vec);
        acti_forget_left.push_back(tmp_vec);
        acti_cell_left.push_back(tmp_vec);
        acti_output_left.push_back(tmp_vec);
        output_h_left.push_back(tmp_vec);

        nonlin_input_right.push_back(tmp_vec);
        nonlin_forget_right.push_back(tmp_vec);
        nonlin_cell_right.push_back(tmp_vec);
        nonlin_output_right.push_back(tmp_vec);
        acti_input_right.push_back(tmp_vec);
        acti_forget_right.push_back(tmp_vec);
        acti_cell_right.push_back(tmp_vec);
        acti_output_right.push_back(tmp_vec);
        output_h_right.push_back(tmp_vec);
        // from left to right
        for(int j = 0; j < T; ++j){
            // for each time slot
            Mat tmp_input, tmp_forget, tmp_cell, tmp_output;
            tmp_input = hLayers[i - 1].U_input_left * output_h_left[i - 1][j];
            tmp_forget = hLayers[i - 1].U_forget_left * output_h_left[i - 1][j];
            tmp_cell = hLayers[i - 1].U_cell_left * output_h_left[i - 1][j];
            tmp_output = hLayers[i - 1].U_output_left * output_h_left[i - 1][j];
            if(j > 0) {
                tmp_input += hLayers[i - 1].W_input_left * output_h_left[i][j - 1];
                tmp_input += hLayers[i - 1].V_input_left -> full * acti_cell_left[i - 1][j - 1];
                tmp_forget += hLayers[i - 1].W_forget_left * output_h_left[i][j - 1];
                tmp_forget += hLayers[i - 1].V_forget_left -> full * acti_cell_left[i - 1][j - 1];
                tmp_cell += hLayers[i - 1].W_cell_left * output_h_left[i][j - 1];
                tmp_output += hLayers[i - 1].W_output_left * output_h_left[i][j - 1];
            }
            if(i > 1){
                tmp_input += hLayers[i - 1].U_input_left * output_h_right[i - 1][j];
                tmp_forget += hLayers[i - 1].U_forget_left * output_h_right[i - 1][j];
                tmp_cell += hLayers[i - 1].U_cell_left * output_h_right[i - 1][j];
                tmp_output += hLayers[i - 1].U_output_left * output_h_right[i - 1][j];
            } 
            tmp_input.copyTo(nonlin_input_left[i - 1][j]);
            tmp_forget.copyTo(nonlin_forget_left[i - 1][j]);
            tmp_cell.copyTo(nonlin_cell_left[i - 1][j]);
            tmp_input = nonLinearity(tmp_input, GATE_NL);
            tmp_forget = nonLinearity(tmp_forget, GATE_NL);
            tmp_cell = nonLinearity(tmp_cell, IO_NL);
            tmp_input.copyTo(acti_input_left[i - 1][j]);
            tmp_forget.copyTo(acti_forget_left[i - 1][j]);
            tmp_cell = tmp_cell.mul(acti_input_left[i - 1][j]);
            if(j > 0){
                tmp_cell += tmp_forget.mul(acti_cell_left[i - 1][j - 1]);
            }
            tmp_cell.copyTo(acti_cell_left[i - 1][j]);
            tmp_output += hLayers[i - 1].V_output_left -> full * tmp_cell;
            tmp_output.copyTo(nonlin_output_left[i - 1][j]);
            tmp_output = nonLinearity(tmp_output, GATE_NL);
            tmp_output.copyTo(acti_output_left[i - 1][j]);
            tmp_output = tmp_output.mul(nonLinearity(tmp_cell, IO_NL));
            tmp_output.copyTo(output_h_left[i][j]);
        }
        // from right to left
        for(int j = T - 1; j >= 0; --j){
            // for each time slot
            Mat tmp_input, tmp_forget, tmp_cell, tmp_output;
            tmp_input = hLayers[i - 1].U_input_right * output_h_right[i - 1][j];
            tmp_forget = hLayers[i - 1].U_forget_right * output_h_right[i - 1][j];
            tmp_cell = hLayers[i - 1].U_cell_right * output_h_right[i - 1][j];
            tmp_output = hLayers[i - 1].U_output_right * output_h_right[i - 1][j];
            if(j < T - 1) {
                tmp_input += hLayers[i - 1].W_input_right * output_h_right[i][j + 1];
                tmp_input += hLayers[i - 1].V_input_right -> full * acti_cell_right[i - 1][j + 1];
                tmp_forget += hLayers[i - 1].W_forget_right * output_h_right[i][j + 1];
                tmp_forget += hLayers[i - 1].V_forget_right -> full * acti_cell_right[i - 1][j + 1];
                tmp_cell += hLayers[i - 1].W_cell_right * output_h_right[i][j + 1];
                tmp_output += hLayers[i - 1].W_output_right * output_h_right[i][j + 1];
            }
            if(i > 1){
                tmp_input += hLayers[i - 1].U_input_right * output_h_left[i - 1][j];
                tmp_forget += hLayers[i - 1].U_forget_right * output_h_left[i - 1][j];
                tmp_cell += hLayers[i - 1].U_cell_right * output_h_left[i - 1][j];
                tmp_output += hLayers[i - 1].U_output_right * output_h_left[i - 1][j];
            }
            tmp_input.copyTo(nonlin_input_right[i - 1][j]);
            tmp_forget.copyTo(nonlin_forget_right[i - 1][j]);
            tmp_cell.copyTo(nonlin_cell_right[i - 1][j]);
            tmp_input = nonLinearity(tmp_input, GATE_NL);
            tmp_forget = nonLinearity(tmp_forget, GATE_NL);
            tmp_cell = nonLinearity(tmp_cell, IO_NL);
            tmp_input.copyTo(acti_input_right[i - 1][j]);
            tmp_forget.copyTo(acti_forget_right[i - 1][j]);
            tmp_cell = tmp_cell.mul(acti_input_right[i - 1][j]);
            if(j < T - 1){
                tmp_cell += tmp_forget.mul(acti_cell_right[i - 1][j + 1]);
            }
            tmp_cell.copyTo(acti_cell_right[i - 1][j]);
            tmp_output += hLayers[i - 1].V_output_right -> full * tmp_cell;
            tmp_output.copyTo(nonlin_output_right[i - 1][j]);
            tmp_output = nonLinearity(tmp_output, GATE_NL);
            tmp_output.copyTo(acti_output_right[i - 1][j]);
            tmp_output = tmp_output.mul(nonLinearity(tmp_cell, IO_NL));
            tmp_output.copyTo(output_h_right[i][j]);
        }
    }
    // softmax layer forward
    std::vector<Mat> p(T);
    for(int i = 0; i < T; ++i){
        Mat M = smr.W_left * output_h_left[output_h_left.size() - 1][i];
        M += smr.W_right * output_h_right[output_h_right.size() - 1][i];
        //M -= repeat(reduce(M, 0, CV_REDUCE_MAX), M.rows, 1);
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
/*
    if(prev_cost != -1.0 && abs(J1) >= 8){
//    if(prev_cost != -1.0 && J1 >= (prev_cost * 5.0)){
        // something's wrong
        cout<<endl;
        cout<<"J1 = "<<J1<<endl;
        cout.flags(ios::scientific);
        cout.precision(6);
        for(int i = 0; i < T; i++){
            cout<<"-----------------------------"<<endl;


            cout<<"nonlin_input  "<<i<<": "<<endl<<" "<<nonlin_input[0][i]<<endl;
            cout<<"acti_input  "<<i<<": "<<endl<<" "<<acti_input[0][i]<<endl;
            cout<<"nonlin_forget  "<<i<<": "<<endl<<" "<<nonlin_forget[0][i]<<endl;
            cout<<"acti_forget  "<<i<<": "<<endl<<" "<<acti_forget[0][i]<<endl;
            cout<<"nonlin_cell  "<<i<<": "<<endl<<" "<<nonlin_cell[0][i]<<endl;

cout<<"******"<<endl;
cout<<nonLinearity(nonlin_cell[0][i])<<endl;
cout<<"******"<<endl;


            cout<<"acti_cell  "<<i<<": "<<endl<<" "<<acti_cell[0][i]<<endl;


            cout<<"output_1_  "<<i<<": "<<endl<<" "<<hLayers[0].U_output * output_h[0][i]<<endl;
            if(i > 0){
                cout<<"output_2_  "<<i<<": "<<endl<<" "<<hLayers[0].W_output * output_h[1][i - 1]<<endl;
            }
            cout<<"output_3_  "<<i<<": "<<endl<<" "<<hLayers[0].V_output -> full * acti_cell[0][i]<<endl;
            cout<<"output"<<i<<": "<<endl<<" "<<acti_output[acti_output.size() - 1][i]<<endl;
            cout<<"groundTruth"<<i<<": "<<endl<<" "<<groundTruth[i]<<endl;
            Mat M = smr.W * output_h[output_h.size() - 1][i];
            cout<<"M--"<<i<<": "<<endl<<" "<<M<<endl;
            //M -= repeat(reduce(M, 0, CV_REDUCE_MAX), M.rows, 1);
            //cout<<"Mred"<<i<<": "<<endl<<" "<<M<<endl;
            M = exp(M);
            cout<<"Mexp"<<i<<": "<<endl<<" "<<M<<endl;
            cout<<"p"<<i<<": "<<endl<<" "<<p[i]<<endl;
            cout<<"J"<<i<<": "<<endl<<" "<<groundTruth[i].mul(log(p[i]))<<endl;
            cout<<"output_h"<<i<<": "<<endl<<" "<<output_h[output_h.size() - 1][i]<<endl;
            cout<<"-----------------------------"<<endl;
        }
        //cout<<output_h[0][0]<<endl;
        //cout<<hLayers[0].V_output -> diagonal<<endl;
        //return false;
        exit(0);
    }else{
        prev_cost = J1;
    }
    //*/
    double J2 = (sum1(pow(smr.W_left, 2.0)) + sum1(pow(smr.W_right, 2.0))) * softmaxConfig.WeightDecay / 2;
    double J3 = 0.0; 
    double J4 = 0.0;
    double J5 = 0.0;
    //cout<<endl;
    for(int hl = 0; hl < hLayers.size(); hl++){
        J3 += (sum1(pow(hLayers[hl].W_input_left, 2.0)) + sum1(pow(hLayers[hl].W_input_right, 2.0))) * hiddenConfig[hl].WeightDecay / 2;
        J3 += (sum1(pow(hLayers[hl].W_forget_left, 2.0)) + sum1(pow(hLayers[hl].W_forget_right, 2.0))) * hiddenConfig[hl].WeightDecay / 2;
        J3 += (sum1(pow(hLayers[hl].W_cell_left, 2.0)) + sum1(pow(hLayers[hl].W_cell_right, 2.0))) * hiddenConfig[hl].WeightDecay / 2;
        J3 += (sum1(pow(hLayers[hl].W_output_left, 2.0)) + sum1(pow(hLayers[hl].W_output_right, 2.0))) * hiddenConfig[hl].WeightDecay / 2;
    }
    for(int hl = 0; hl < hLayers.size(); hl++){
        J4 += (sum1(pow(hLayers[hl].U_input_left, 2.0)) + sum1(pow(hLayers[hl].U_input_right, 2.0))) * hiddenConfig[hl].WeightDecay / 2;
        J4 += (sum1(pow(hLayers[hl].U_forget_left, 2.0)) + sum1(pow(hLayers[hl].U_forget_right, 2.0))) * hiddenConfig[hl].WeightDecay / 2;
        J4 += (sum1(pow(hLayers[hl].U_cell_left, 2.0)) + sum1(pow(hLayers[hl].U_cell_right, 2.0))) * hiddenConfig[hl].WeightDecay / 2;
        J4 += (sum1(pow(hLayers[hl].U_output_left, 2.0)) + sum1(pow(hLayers[hl].U_output_right, 2.0))) * hiddenConfig[hl].WeightDecay / 2;
    }
    
    for(int hl = 0; hl < hLayers.size(); hl++){
        J5 += (sum1(pow(hLayers[hl].V_input_left -> diagonal, 2.0)) + sum1(pow(hLayers[hl].V_input_right -> diagonal, 2.0))) * hiddenConfig[hl].WeightDecay / 2;
        J5 += (sum1(pow(hLayers[hl].V_forget_left -> diagonal, 2.0)) + sum1(pow(hLayers[hl].V_forget_right -> diagonal, 2.0))) * hiddenConfig[hl].WeightDecay / 2;
        J5 += (sum1(pow(hLayers[hl].V_output_left -> diagonal, 2.0)) + sum1(pow(hLayers[hl].V_output_right -> diagonal, 2.0))) * hiddenConfig[hl].WeightDecay / 2;
    }//*/
    smr.cost = J1 + J2 + J3 + J4 + J5;
    if(!is_gradient_checking) 
        cout<<", J1 = "<<J1<<", J2 = "<<J2<<", J3 = "<<J3<<", J4 = "<<J4<<", J5 = "<<J5<<", Cost = "<<smr.cost<<endl;

    // softmax layer backward
    Mat tmp, tmp2;
    tmp = - (groundTruth[0] - p[0]) * output_h_left[output_h_left.size() - 1][0].t();
    tmp2 = pow((groundTruth[0] - p[0]), 2.0) * pow(output_h_left[output_h_left.size() - 1][0].t(), 2.0);
    for(int i = 1; i < T; ++i){
        tmp += - (groundTruth[i] - p[i]) * output_h_left[output_h_left.size() - 1][i].t();
        tmp2 += pow((groundTruth[i] - p[i]), 2.0) * pow(output_h_left[output_h_left.size() - 1][i].t(), 2.0);
    }
    smr.Wgrad_left =  tmp / nSamples + softmaxConfig.WeightDecay * smr.W_left;
    smr.Wd2_left = tmp2 / nSamples + softmaxConfig.WeightDecay;

    tmp = - (groundTruth[0] - p[0]) * output_h_right[output_h_right.size() - 1][0].t();
    tmp2 = pow((groundTruth[0] - p[0]), 2.0) * pow(output_h_right[output_h_right.size() - 1][0].t(), 2.0);
    for(int i = 1; i < T; ++i){
        tmp += - (groundTruth[i] - p[i]) * output_h_right[output_h_right.size() - 1][i].t();
        tmp2 += pow((groundTruth[i] - p[i]), 2.0) * pow(output_h_right[output_h_right.size() - 1][i].t(), 2.0);
    }
    smr.Wgrad_right =  tmp / nSamples + softmaxConfig.WeightDecay * smr.W_right;
    smr.Wd2_right = tmp2 / nSamples + softmaxConfig.WeightDecay;

    // hidden layer backward
    std::vector<std::vector<Mat> > delta_input_left;
    std::vector<std::vector<Mat> > deltad2_input_left;
    std::vector<std::vector<Mat> > delta_forget_left;
    std::vector<std::vector<Mat> > deltad2_forget_left;
    std::vector<std::vector<Mat> > delta_cell_left;
    std::vector<std::vector<Mat> > deltad2_cell_left;
    std::vector<std::vector<Mat> > delta_output_left;
    std::vector<std::vector<Mat> > deltad2_output_left;
    std::vector<std::vector<Mat> > epsilon_output_left;
    std::vector<std::vector<Mat> > epsilond2_output_left;
    std::vector<std::vector<Mat> > epsilon_state_left;
    std::vector<std::vector<Mat> > epsilond2_state_left;

    std::vector<std::vector<Mat> > delta_input_right;
    std::vector<std::vector<Mat> > deltad2_input_right;
    std::vector<std::vector<Mat> > delta_forget_right;
    std::vector<std::vector<Mat> > deltad2_forget_right;
    std::vector<std::vector<Mat> > delta_cell_right;
    std::vector<std::vector<Mat> > deltad2_cell_right;
    std::vector<std::vector<Mat> > delta_output_right;
    std::vector<std::vector<Mat> > deltad2_output_right;
    std::vector<std::vector<Mat> > epsilon_output_right;
    std::vector<std::vector<Mat> > epsilond2_output_right;
    std::vector<std::vector<Mat> > epsilon_state_right;
    std::vector<std::vector<Mat> > epsilond2_state_right;
    for(int i = 0; i < output_h_left.size(); i++){
        delta_input_left.push_back(tmp_vec);
        deltad2_input_left.push_back(tmp_vec);
        delta_forget_left.push_back(tmp_vec);
        deltad2_forget_left.push_back(tmp_vec);
        delta_cell_left.push_back(tmp_vec);
        deltad2_cell_left.push_back(tmp_vec);
        delta_output_left.push_back(tmp_vec);
        deltad2_output_left.push_back(tmp_vec);
        epsilon_output_left.push_back(tmp_vec);
        epsilond2_output_left.push_back(tmp_vec);
        epsilon_state_left.push_back(tmp_vec);
        epsilond2_state_left.push_back(tmp_vec);

        delta_input_right.push_back(tmp_vec);
        deltad2_input_right.push_back(tmp_vec);
        delta_forget_right.push_back(tmp_vec);
        deltad2_forget_right.push_back(tmp_vec);
        delta_cell_right.push_back(tmp_vec);
        deltad2_cell_right.push_back(tmp_vec);
        delta_output_right.push_back(tmp_vec);
        deltad2_output_right.push_back(tmp_vec);
        epsilon_output_right.push_back(tmp_vec);
        epsilond2_output_right.push_back(tmp_vec);
        epsilon_state_right.push_back(tmp_vec);
        epsilond2_state_right.push_back(tmp_vec);
    }
    // Last hidden layer
    // Do BPTT backward pass for the forward hidden layer
    for(int i = T - 1; i >= 0; --i){
        // cell output
        tmp = -smr.W_left.t() * (groundTruth[i] - p[i]);
        tmp2 = pow(smr.W_left.t(), 2.0) * pow((groundTruth[i] - p[i]), 2.0);
        if(i < T - 1){            
            tmp += hLayers[hLayers.size() - 1].W_cell_left.t() * delta_cell_left[delta_cell_left.size() - 1][i + 1];
            tmp += hLayers[hLayers.size() - 1].W_input_left.t() * delta_input_left[delta_input_left.size() - 1][i + 1];
            tmp += hLayers[hLayers.size() - 1].W_forget_left.t() * delta_forget_left[delta_forget_left.size() - 1][i + 1];
            tmp += hLayers[hLayers.size() - 1].W_output_left.t() * delta_output_left[delta_output_left.size() - 1][i + 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].W_cell_left.t(), 2.0) * deltad2_cell_left[deltad2_cell_left.size() - 1][i + 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].W_input_left.t(), 2.0) * deltad2_input_left[deltad2_input_left.size() - 1][i + 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].W_forget_left.t(), 2.0) * deltad2_forget_left[deltad2_forget_left.size() - 1][i + 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].W_output_left.t(), 2.0) * deltad2_output_left[deltad2_output_left.size() - 1][i + 1];
        }
        tmp.copyTo(epsilon_output_left[epsilon_output_left.size() - 1][i]);
        tmp2.copyTo(epsilond2_output_left[epsilond2_output_left.size() - 1][i]);
        // output gates
        tmp = tmp.mul(nonLinearity(acti_cell_left[acti_cell_left.size() - 1][i], IO_NL));
        tmp = tmp.mul(dnonLinearity(nonlin_output_left[nonlin_output_left.size() - 1][i], GATE_NL));
        tmp2 = tmp2.mul(pow(nonLinearity(acti_cell_left[acti_cell_left.size() - 1][i], IO_NL), 2.0));
        tmp2 = tmp2.mul(pow(dnonLinearity(nonlin_output_left[nonlin_output_left.size() - 1][i], GATE_NL), 2.0));
        tmp.copyTo(delta_output_left[delta_output_left.size() - 1][i]);
        tmp2.copyTo(deltad2_output_left[deltad2_output_left.size() - 1][i]);
        // states
        tmp = acti_output_left[acti_output_left.size() - 1][i].mul(dnonLinearity(acti_cell_left[acti_cell_left.size() - 1][i], IO_NL));
        tmp = tmp.mul(epsilon_output_left[epsilon_output_left.size() - 1][i]);
        tmp2 = pow(acti_output_left[acti_output_left.size() - 1][i], 2.0).mul(pow(dnonLinearity(acti_cell_left[acti_cell_left.size() - 1][i], IO_NL), 2.0));
        tmp2 = tmp2.mul(epsilond2_output_left[epsilond2_output_left.size() - 1][i]);
        if(i < T - 1){
            tmp += acti_forget_left[acti_forget_left.size() - 1][i + 1].mul(epsilon_state_left[epsilon_state_left.size() - 1][i + 1]);
            tmp += hLayers[hLayers.size() - 1].V_input_left -> full.t() * delta_input_left[delta_input_left.size() - 1][i + 1];
            tmp += hLayers[hLayers.size() - 1].V_forget_left -> full.t() * delta_forget_left[delta_forget_left.size() - 1][i + 1];
            tmp2 += pow(acti_forget_left[acti_forget_left.size() - 1][i + 1], 2.0).mul(epsilond2_state_left[epsilond2_state_left.size() - 1][i + 1]);
            tmp2 += pow(hLayers[hLayers.size() - 1].V_input_left -> full.t(), 2.0) * deltad2_input_left[deltad2_input_left.size() - 1][i + 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].V_forget_left -> full.t(), 2.0) * deltad2_forget_left[deltad2_forget_left.size() - 1][i + 1];
        }
        tmp += hLayers[hLayers.size() - 1].V_output_left -> full.t() * delta_output_left[delta_output_left.size() - 1][i];
        tmp2 += pow(hLayers[hLayers.size() - 1].V_output_left -> full.t(), 2.0) * deltad2_output_left[deltad2_output_left.size() - 1][i];
        tmp.copyTo(epsilon_state_left[epsilon_state_left.size() - 1][i]);
        tmp2.copyTo(epsilond2_state_left[epsilond2_state_left.size() - 1][i]);
        // cells
        tmp = acti_input_left[acti_input_left.size() - 1][i].mul(dnonLinearity(nonlin_cell_left[nonlin_cell_left.size() - 1][i], IO_NL));
        tmp = tmp.mul(epsilon_state_left[epsilon_state_left.size() - 1][i]);
        tmp2 = pow(acti_input_left[acti_input_left.size() - 1][i], 2.0).mul(pow(dnonLinearity(nonlin_cell_left[nonlin_cell_left.size() - 1][i], IO_NL), 2.0));
        tmp2 = tmp2.mul(epsilond2_state_left[epsilond2_state_left.size() - 1][i]);
        tmp.copyTo(delta_cell_left[delta_cell_left.size() - 1][i]);
        tmp2.copyTo(deltad2_cell_left[deltad2_cell_left.size() - 1][i]);
        // forget gates
        if(i > 0){
            tmp = acti_cell_left[acti_cell_left.size() - 1][i - 1].mul(epsilon_state_left[epsilon_state_left.size() - 1][i]);
            tmp = tmp.mul(dnonLinearity(nonlin_forget_left[nonlin_forget_left.size() - 1][i], GATE_NL));
            tmp2 = pow(acti_cell_left[acti_cell_left.size() - 1][i - 1], 2.0).mul(epsilond2_state_left[epsilond2_state_left.size() - 1][i]);
            tmp2 = tmp2.mul(pow(dnonLinearity(nonlin_forget_left[nonlin_forget_left.size() - 1][i], GATE_NL), 2.0));
        }else{
            tmp = Mat::zeros(nonlin_forget_left[nonlin_forget_left.size() - 1][i].size(), CV_64FC1);
            tmp2 = Mat::zeros(nonlin_forget_left[nonlin_forget_left.size() - 1][i].size(), CV_64FC1);
        }
        tmp.copyTo(delta_forget_left[delta_forget_left.size() - 1][i]);
        tmp2.copyTo(deltad2_forget_left[deltad2_forget_left.size() - 1][i]);
        // input gates
        tmp = epsilon_state_left[epsilon_state_left.size() - 1][i].mul(nonLinearity(nonlin_cell_left[nonlin_cell_left.size() - 1][i], IO_NL));
        tmp = tmp.mul(dnonLinearity(nonlin_input_left[nonlin_input_left.size() - 1][i], GATE_NL));
        tmp2 = epsilond2_state_left[epsilond2_state_left.size() - 1][i].mul(pow(nonLinearity(nonlin_cell_left[nonlin_cell_left.size() - 1][i], IO_NL), 2.0));
        tmp2 = tmp2.mul(pow(dnonLinearity(nonlin_input_left[nonlin_input_left.size() - 1][i], GATE_NL), 2.0));
        tmp.copyTo(delta_input_left[delta_input_left.size() - 1][i]);
        tmp2.copyTo(deltad2_input_left[deltad2_input_left.size() - 1][i]);
    }
    // Do BPTT backward pass for the backward hidden layer
    for(int i = 0; i < T; i++){
        // cell output
        tmp = -smr.W_right.t() * (groundTruth[i] - p[i]);
        tmp2 = pow(smr.W_right.t(), 2.0) * pow((groundTruth[i] - p[i]), 2.0);
        if(i > 0){            
            tmp += hLayers[hLayers.size() - 1].W_cell_right.t() * delta_cell_right[delta_cell_right.size() - 1][i - 1];
            tmp += hLayers[hLayers.size() - 1].W_input_right.t() * delta_input_right[delta_input_right.size() - 1][i - 1];
            tmp += hLayers[hLayers.size() - 1].W_forget_right.t() * delta_forget_right[delta_forget_right.size() - 1][i - 1];
            tmp += hLayers[hLayers.size() - 1].W_output_right.t() * delta_output_right[delta_output_right.size() - 1][i - 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].W_cell_right.t(), 2.0) * deltad2_cell_right[deltad2_cell_right.size() - 1][i - 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].W_input_right.t(), 2.0) * deltad2_input_right[deltad2_input_right.size() - 1][i - 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].W_forget_right.t(), 2.0) * deltad2_forget_right[deltad2_forget_right.size() - 1][i - 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].W_output_right.t(), 2.0) * deltad2_output_right[deltad2_output_right.size() - 1][i - 1];
        }
        tmp.copyTo(epsilon_output_right[epsilon_output_right.size() - 1][i]);
        tmp2.copyTo(epsilond2_output_right[epsilond2_output_right.size() - 1][i]);
        // output gates
        tmp = tmp.mul(nonLinearity(acti_cell_right[acti_cell_right.size() - 1][i], IO_NL));
        tmp = tmp.mul(dnonLinearity(nonlin_output_right[nonlin_output_right.size() - 1][i], GATE_NL));
        tmp2 = tmp2.mul(pow(nonLinearity(acti_cell_right[acti_cell_right.size() - 1][i], IO_NL), 2.0));
        tmp2 = tmp2.mul(pow(dnonLinearity(nonlin_output_right[nonlin_output_right.size() - 1][i], GATE_NL), 2.0));
        tmp.copyTo(delta_output_right[delta_output_right.size() - 1][i]);
        tmp2.copyTo(deltad2_output_right[deltad2_output_right.size() - 1][i]);
        // states
        tmp = acti_output_right[acti_output_right.size() - 1][i].mul(dnonLinearity(acti_cell_right[acti_cell_right.size() - 1][i], IO_NL));
        tmp = tmp.mul(epsilon_output_right[epsilon_output_right.size() - 1][i]);
        tmp2 = pow(acti_output_right[acti_output_right.size() - 1][i], 2.0).mul(pow(dnonLinearity(acti_cell_right[acti_cell_right.size() - 1][i], IO_NL), 2.0));
        tmp2 = tmp2.mul(epsilond2_output_right[epsilond2_output_right.size() - 1][i]);
        if(i > 0){
            tmp += acti_forget_right[acti_forget_right.size() - 1][i - 1].mul(epsilon_state_right[epsilon_state_right.size() - 1][i - 1]);
            tmp += hLayers[hLayers.size() - 1].V_input_right -> full.t() * delta_input_right[delta_input_right.size() - 1][i - 1];
            tmp += hLayers[hLayers.size() - 1].V_forget_right -> full.t() * delta_forget_right[delta_forget_right.size() - 1][i - 1];
            tmp2 += pow(acti_forget_right[acti_forget_right.size() - 1][i - 1], 2.0).mul(epsilond2_state_right[epsilond2_state_right.size() - 1][i - 1]);
            tmp2 += pow(hLayers[hLayers.size() - 1].V_input_right -> full.t(), 2.0) * deltad2_input_right[deltad2_input_right.size() - 1][i - 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].V_forget_right -> full.t(), 2.0) * deltad2_forget_right[deltad2_forget_right.size() - 1][i - 1];
        }
        tmp += hLayers[hLayers.size() - 1].V_output_right -> full.t() * delta_output_right[delta_output_right.size() - 1][i];
        tmp2 += pow(hLayers[hLayers.size() - 1].V_output_right -> full.t(), 2.0) * deltad2_output_right[deltad2_output_right.size() - 1][i];
        tmp.copyTo(epsilon_state_right[epsilon_state_right.size() - 1][i]);
        tmp2.copyTo(epsilond2_state_right[epsilond2_state_right.size() - 1][i]);
        // cells
        tmp = acti_input_right[acti_input_right.size() - 1][i].mul(dnonLinearity(nonlin_cell_right[nonlin_cell_right.size() - 1][i], IO_NL));
        tmp = tmp.mul(epsilon_state_right[epsilon_state_right.size() - 1][i]);
        tmp2 = pow(acti_input_right[acti_input_right.size() - 1][i], 2.0).mul(pow(dnonLinearity(nonlin_cell_right[nonlin_cell_right.size() - 1][i], IO_NL), 2.0));
        tmp2 = tmp2.mul(epsilond2_state_right[epsilond2_state_right.size() - 1][i]);
        tmp.copyTo(delta_cell_right[delta_cell_right.size() - 1][i]);
        tmp2.copyTo(deltad2_cell_right[deltad2_cell_right.size() - 1][i]);
        // forget gates
        if(i < T - 1){
            tmp = acti_cell_right[acti_cell_right.size() - 1][i + 1].mul(epsilon_state_right[epsilon_state_right.size() - 1][i]);
            tmp = tmp.mul(dnonLinearity(nonlin_forget_right[nonlin_forget_right.size() - 1][i], GATE_NL));
            tmp2 = pow(acti_cell_right[acti_cell_right.size() - 1][i + 1], 2.0).mul(epsilond2_state_right[epsilond2_state_right.size() - 1][i]);
            tmp2 = tmp2.mul(pow(dnonLinearity(nonlin_forget_right[nonlin_forget_right.size() - 1][i], GATE_NL), 2.0));
        }else{
            tmp = Mat::zeros(nonlin_forget_right[nonlin_forget_right.size() - 1][i].size(), CV_64FC1);
            tmp2 = Mat::zeros(nonlin_forget_right[nonlin_forget_right.size() - 1][i].size(), CV_64FC1);
        }
        tmp.copyTo(delta_forget_right[delta_forget_right.size() - 1][i]);
        tmp2.copyTo(deltad2_forget_right[deltad2_forget_right.size() - 1][i]);
        // input gates
        tmp = epsilon_state_right[epsilon_state_right.size() - 1][i].mul(nonLinearity(nonlin_cell_right[nonlin_cell_right.size() - 1][i], IO_NL));
        tmp = tmp.mul(dnonLinearity(nonlin_input_right[nonlin_input_right.size() - 1][i], GATE_NL));
        tmp2 = epsilond2_state_right[epsilond2_state_right.size() - 1][i].mul(pow(nonLinearity(nonlin_cell_right[nonlin_cell_right.size() - 1][i], IO_NL), 2.0));
        tmp2 = tmp2.mul(pow(dnonLinearity(nonlin_input_right[nonlin_input_right.size() - 1][i], GATE_NL), 2.0));
        tmp.copyTo(delta_input_right[delta_input_right.size() - 1][i]);
        tmp2.copyTo(deltad2_input_right[deltad2_input_right.size() - 1][i]);
    }    
    // hidden layers
    for(int i = delta_input_left.size() - 2; i > 0; --i){
        // Do BPTT backward pass for the forward hidden layer
        for(int j = T - 1; j >= 0; --j){
            // cell output
            tmp = hLayers[i].U_cell_left.t() * delta_cell_left[i + 1][j];
            tmp += hLayers[i].U_input_left.t() * delta_input_left[i + 1][j];
            tmp += hLayers[i].U_forget_left.t() * delta_forget_left[i + 1][j];
            tmp += hLayers[i].U_output_left.t() * delta_output_left[i + 1][j];
            tmp2 = pow(hLayers[i].U_cell_left.t(), 2.0) * deltad2_cell_left[i + 1][j];
            tmp2 += pow(hLayers[i].U_input_left.t(), 2.0) * deltad2_input_left[i + 1][j];
            tmp2 += pow(hLayers[i].U_forget_left.t(), 2.0) * deltad2_forget_left[i + 1][j];
            tmp2 += pow(hLayers[i].U_output_left.t(), 2.0) * deltad2_output_left[i + 1][j];
            if(j < T - 1){
                tmp += hLayers[i - 1].W_cell_left.t() * delta_cell_left[i][j + 1];
                tmp += hLayers[i - 1].W_input_left.t() * delta_input_left[i][j + 1];
                tmp += hLayers[i - 1].W_forget_left.t() * delta_forget_left[i][j + 1];
                tmp += hLayers[i - 1].W_output_left.t() * delta_output_left[i][j + 1];
                tmp2 += pow(hLayers[i - 1].W_cell_left.t(), 2.0) * deltad2_cell_left[i][j + 1];
                tmp2 += pow(hLayers[i - 1].W_input_left.t(), 2.0) * deltad2_input_left[i][j + 1];
                tmp2 += pow(hLayers[i - 1].W_forget_left.t(), 2.0) * deltad2_forget_left[i][j + 1];
                tmp2 += pow(hLayers[i - 1].W_output_left.t(), 2.0) * deltad2_output_left[i][j + 1];
            } 
            tmp += hLayers[i].U_cell_right.t() * delta_cell_right[i + 1][j];
            tmp += hLayers[i].U_input_right.t() * delta_input_right[i + 1][j];
            tmp += hLayers[i].U_forget_right.t() * delta_forget_right[i + 1][j];
            tmp += hLayers[i].U_output_right.t() * delta_output_right[i + 1][j];
            tmp2 += pow(hLayers[i].U_cell_right.t(), 2.0) * deltad2_cell_right[i + 1][j];
            tmp2 += pow(hLayers[i].U_input_right.t(), 2.0) * deltad2_input_right[i + 1][j];
            tmp2 += pow(hLayers[i].U_forget_right.t(), 2.0) * deltad2_forget_right[i + 1][j];
            tmp2 += pow(hLayers[i].U_output_right.t(), 2.0) * deltad2_output_right[i + 1][j];
            tmp.copyTo(epsilon_output_left[i][j]);
            tmp2.copyTo(epsilond2_output_left[i][j]);
            // output gates
            tmp = tmp.mul(nonLinearity(acti_cell_left[i - 1][j], IO_NL));
            tmp = tmp.mul(dnonLinearity(nonlin_output_left[i - 1][j], GATE_NL));
            tmp2 = tmp2.mul(pow(nonLinearity(acti_cell_left[i - 1][j], IO_NL), 2.0));
            tmp2 = tmp2.mul(pow(dnonLinearity(nonlin_output_left[i - 1][j], GATE_NL), 2.0));
            tmp.copyTo(delta_output_left[i][j]);
            tmp2.copyTo(deltad2_output_left[i][j]);
            // states
            tmp = acti_output_left[i - 1][j].mul(dnonLinearity(acti_cell_left[i - 1][j], IO_NL));
            tmp = tmp.mul(epsilon_output_left[i][j]);
            tmp2 = pow(acti_output_left[i - 1][j], 2.0).mul(pow(dnonLinearity(acti_cell_left[i - 1][j], IO_NL), 2.0));
            tmp2 = tmp2.mul(epsilond2_output_left[i][j]);
            if(j < T - 1){
                tmp += acti_forget_left[i - 1][j + 1].mul(epsilon_state_left[i][j + 1]);
                tmp += hLayers[i - 1].V_input_left -> full.t() * delta_input_left[i][j + 1];
                tmp += hLayers[i - 1].V_forget_left -> full.t() * delta_forget_left[i][j + 1];
                tmp2 += pow(acti_forget_left[i - 1][j + 1], 2.0).mul(epsilond2_state_left[i][j + 1]);
                tmp2 += pow(hLayers[i - 1].V_input_left -> full.t(), 2.0) * deltad2_input_left[i][j + 1];
                tmp2 += pow(hLayers[i - 1].V_forget_left -> full.t(), 2.0) * deltad2_forget_left[i][j + 1];
            }
            tmp += hLayers[i - 1].V_output_left -> full.t() * delta_output_left[i][j];
            tmp2 += pow(hLayers[i - 1].V_output_left -> full.t(), 2.0) * deltad2_output_left[i][j];
            tmp.copyTo(epsilon_state_left[i][j]);
            tmp2.copyTo(epsilond2_state_left[i][j]);
            // cells
            tmp = acti_input_left[i - 1][j].mul(dnonLinearity(nonlin_cell_left[i - 1][j], IO_NL));
            tmp = tmp.mul(epsilon_state_left[i][j]);
            tmp2 = pow(acti_input_left[i - 1][j], 2.0).mul(pow(dnonLinearity(nonlin_cell_left[i - 1][j], IO_NL), 2.0));
            tmp2 = tmp2.mul(epsilond2_state_left[i][j]);
            tmp.copyTo(delta_cell_left[i][j]);
            tmp2.copyTo(deltad2_cell_left[i][j]);
            // forget gates
            if(j > 0){
                tmp = acti_cell_left[i - 1][j - 1].mul(epsilon_state_left[i][j]);
                tmp = tmp.mul(dnonLinearity(nonlin_forget_left[i - 1][j], GATE_NL));
                tmp2 = pow(acti_cell_left[i - 1][j - 1], 2.0).mul(epsilond2_state_left[i][j]);
                tmp2 = tmp2.mul(pow(dnonLinearity(nonlin_forget_left[i - 1][j], GATE_NL), 2.0));
            }else{
                tmp = Mat::zeros(nonlin_forget_left[i - 1][j].size(), CV_64FC1);
                tmp2 = Mat::zeros(nonlin_forget_left[i - 1][j].size(), CV_64FC1);
            }
            tmp.copyTo(delta_forget_left[i][j]);
            tmp2.copyTo(deltad2_forget_left[i][j]);
            // input gates
            tmp = epsilon_state_left[i][j].mul(nonLinearity(nonlin_cell_left[i - 1][j], IO_NL));
            tmp = tmp.mul(dnonLinearity(nonlin_input_left[i - 1][j], GATE_NL));
            tmp2 = epsilond2_state_left[i][j].mul(pow(nonLinearity(nonlin_cell_left[i - 1][j], IO_NL), 2.0));
            tmp2 = tmp2.mul(pow(dnonLinearity(nonlin_input_left[i - 1][j], GATE_NL), 2.0));
            tmp.copyTo(delta_input_left[i][j]);
            tmp2.copyTo(deltad2_input_left[i][j]);
        }
        // Do BPTT backward pass for the backward hidden layer
        for(int j = 0; j < T; ++j){
            // cell  output
            tmp = hLayers[i].U_cell_right.t() * delta_cell_right[i + 1][j];
            tmp += hLayers[i].U_input_right.t() * delta_input_right[i + 1][j];
            tmp += hLayers[i].U_forget_right.t() * delta_forget_right[i + 1][j];
            tmp += hLayers[i].U_output_right.t() * delta_output_right[i + 1][j];
            tmp2 = pow(hLayers[i].U_cell_right.t(), 2.0) * deltad2_cell_right[i + 1][j];
            tmp2 += pow(hLayers[i].U_input_right.t(), 2.0) * deltad2_input_right[i + 1][j];
            tmp2 += pow(hLayers[i].U_forget_right.t(), 2.0) * deltad2_forget_right[i + 1][j];
            tmp2 += pow(hLayers[i].U_output_right.t(), 2.0) * deltad2_output_right[i + 1][j];
            if(j > 0){
                tmp += hLayers[i - 1].W_cell_right.t() * delta_cell_right[i][j - 1];
                tmp += hLayers[i - 1].W_input_right.t() * delta_input_right[i][j - 1];
                tmp += hLayers[i - 1].W_forget_right.t() * delta_forget_right[i][j - 1];
                tmp += hLayers[i - 1].W_output_right.t() * delta_output_right[i][j - 1];
                tmp2 += pow(hLayers[i - 1].W_cell_right.t(), 2.0) * deltad2_cell_right[i][j - 1];
                tmp2 += pow(hLayers[i - 1].W_input_right.t(), 2.0) * deltad2_input_right[i][j - 1];
                tmp2 += pow(hLayers[i - 1].W_forget_right.t(), 2.0) * deltad2_forget_right[i][j - 1];
                tmp2 += pow(hLayers[i - 1].W_output_right.t(), 2.0) * deltad2_output_right[i][j - 1];
            }
            tmp += hLayers[i].U_cell_left.t() * delta_cell_left[i + 1][j];
            tmp += hLayers[i].U_input_left.t() * delta_input_left[i + 1][j];
            tmp += hLayers[i].U_forget_left.t() * delta_forget_left[i + 1][j];
            tmp += hLayers[i].U_output_left.t() * delta_output_left[i + 1][j];
            tmp2 += pow(hLayers[i].U_cell_left.t(), 2.0) * deltad2_cell_left[i + 1][j];
            tmp2 += pow(hLayers[i].U_input_left.t(), 2.0) * deltad2_input_left[i + 1][j];
            tmp2 += pow(hLayers[i].U_forget_left.t(), 2.0) * deltad2_forget_left[i + 1][j];
            tmp2 += pow(hLayers[i].U_output_left.t(), 2.0) * deltad2_output_left[i + 1][j];
            tmp.copyTo(epsilon_output_right[i][j]);
            tmp2.copyTo(epsilond2_output_right[i][j]);
            // output gates
            tmp = tmp.mul(nonLinearity(acti_cell_right[i - 1][j], IO_NL));
            tmp = tmp.mul(dnonLinearity(nonlin_output_right[i - 1][j], GATE_NL));
            tmp2 = tmp2.mul(pow(nonLinearity(acti_cell_right[i - 1][j], IO_NL), 2.0));
            tmp2 = tmp2.mul(pow(dnonLinearity(nonlin_output_right[i - 1][j], GATE_NL), 2.0));
            tmp.copyTo(delta_output_right[i][j]);
            tmp2.copyTo(deltad2_output_right[i][j]);
            // states
            tmp = acti_output_right[i - 1][j].mul(dnonLinearity(acti_cell_right[i - 1][j], IO_NL));
            tmp = tmp.mul(epsilon_output_right[i][j]);
            tmp2 = pow(acti_output_right[i - 1][j], 2.0).mul(pow(dnonLinearity(acti_cell_right[i - 1][j], IO_NL), 2.0));
            tmp2 = tmp2.mul(epsilond2_output_right[i][j]);
            if(j > 0){
                tmp += acti_forget_right[i - 1][j - 1].mul(epsilon_state_right[i][j - 1]);
                tmp += hLayers[i - 1].V_input_right -> full.t() * delta_input_right[i][j - 1];
                tmp += hLayers[i - 1].V_forget_right -> full.t() * delta_forget_right[i][j - 1];
                tmp2 += pow(acti_forget_right[i - 1][j - 1], 2.0).mul(epsilond2_state_right[i][j - 1]);
                tmp2 += pow(hLayers[i - 1].V_input_right -> full.t(), 2.0) * deltad2_input_right[i][j - 1];
                tmp2 += pow(hLayers[i - 1].V_forget_right -> full.t(), 2.0) * deltad2_forget_right[i][j - 1];
            }
            tmp += hLayers[i - 1].V_output_right -> full.t() * delta_output_right[i][j];
            tmp2 += pow(hLayers[i - 1].V_output_right -> full.t(), 2.0) * deltad2_output_right[i][j];
            tmp.copyTo(epsilon_state_right[i][j]);
            tmp2.copyTo(epsilond2_state_right[i][j]);
            // cells
            tmp = acti_input_right[i - 1][j].mul(dnonLinearity(nonlin_cell_right[i - 1][j], IO_NL));
            tmp = tmp.mul(epsilon_state_right[i][j]);
            tmp2 = pow(acti_input_right[i - 1][j], 2.0).mul(pow(dnonLinearity(nonlin_cell_right[i - 1][j], IO_NL), 2.0));
            tmp2 = tmp2.mul(epsilond2_state_right[i][j]);
            tmp.copyTo(delta_cell_right[i][j]);
            tmp2.copyTo(deltad2_cell_right[i][j]);
            // forget gates
            if(j < T - 1){
                tmp = acti_cell_right[i - 1][j + 1].mul(epsilon_state_right[i][j]);
                tmp = tmp.mul(dnonLinearity(nonlin_forget_right[i - 1][j], GATE_NL));
                tmp2 = pow(acti_cell_right[i - 1][j + 1], 2.0).mul(epsilond2_state_right[i][j]);
                tmp2 = tmp2.mul(pow(dnonLinearity(nonlin_forget_right[i - 1][j], GATE_NL), 2.0));
            }else{
                tmp = Mat::zeros(nonlin_forget_right[i - 1][j].size(), CV_64FC1);
                tmp2 = Mat::zeros(nonlin_forget_right[i - 1][j].size(), CV_64FC1);
            }
            tmp.copyTo(delta_forget_right[i][j]);
            tmp2.copyTo(deltad2_forget_right[i][j]);
            // input gates
            tmp = epsilon_state_right[i][j].mul(nonLinearity(nonlin_cell_right[i - 1][j], IO_NL));
            tmp = tmp.mul(dnonLinearity(nonlin_input_right[i - 1][j], GATE_NL));
            tmp2 = epsilond2_state_right[i][j].mul(pow(nonLinearity(nonlin_cell_right[i - 1][j], IO_NL), 2.0));
            tmp2 = tmp2.mul(pow(dnonLinearity(nonlin_input_right[i - 1][j], GATE_NL), 2.0));
            tmp.copyTo(delta_input_right[i][j]);
            tmp2.copyTo(deltad2_input_right[i][j]);
        }
    }
//*/
    for(int i = hiddenConfig.size() - 1; i >= 0; i--){
        // forward part.
        // U
        tmp = delta_input_left[i + 1][0] * output_h_left[i][0].t();
        tmp2 = deltad2_input_left[i + 1][0] * pow(output_h_left[i][0].t(), 2.0);
        if(i > 0){
            tmp += delta_input_left[i + 1][0] * output_h_right[i][0].t();
            tmp2 += deltad2_input_left[i + 1][0] * pow(output_h_right[i][0].t(), 2.0);
        }
        for(int j = 1; j < T; ++j){
            tmp += delta_input_left[i + 1][j] * output_h_left[i][j].t();
            tmp2 += deltad2_input_left[i + 1][j] * pow(output_h_left[i][j].t(), 2.0);
            if(i > 0){
                tmp += delta_input_left[i + 1][j] * output_h_right[i][j].t();
                tmp2 += deltad2_input_left[i + 1][j] * pow(output_h_right[i][j].t(), 2.0);
            }
        }
        hLayers[i].Ugrad_input_left = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].U_input_left;
        hLayers[i].Ud2_input_left = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_forget_left[i + 1][0] * output_h_left[i][0].t();
        tmp2 = deltad2_forget_left[i + 1][0] * pow(output_h_left[i][0].t(), 2.0);
        if(i > 0){
            tmp += delta_forget_left[i + 1][0] * output_h_right[i][0].t();
            tmp2 += deltad2_forget_left[i + 1][0] * pow(output_h_right[i][0].t(), 2.0);
        }
        for(int j = 1; j < T; ++j){
            tmp += delta_forget_left[i + 1][j] * output_h_left[i][j].t();
            tmp2 += deltad2_forget_left[i + 1][j] * pow(output_h_left[i][j].t(), 2.0);
            if(i > 0){
                tmp += delta_forget_left[i + 1][j] * output_h_right[i][j].t();
                tmp2 += deltad2_forget_left[i + 1][j] * pow(output_h_right[i][j].t(), 2.0);
            }
        }
        hLayers[i].Ugrad_forget_left = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].U_forget_left;
        hLayers[i].Ud2_forget_left = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_cell_left[i + 1][0] * output_h_left[i][0].t();
        tmp2 = deltad2_cell_left[i + 1][0] * pow(output_h_left[i][0].t(), 2.0);
        if(i > 0){
            tmp += delta_cell_left[i + 1][0] * output_h_right[i][0].t();
            tmp2 += deltad2_cell_left[i + 1][0] * pow(output_h_right[i][0].t(), 2.0);
        }
        for(int j = 1; j < T; ++j){
            tmp += delta_cell_left[i + 1][j] * output_h_left[i][j].t();
            tmp2 += deltad2_cell_left[i + 1][j] * pow(output_h_left[i][j].t(), 2.0);
            if(i > 0){
                tmp += delta_cell_left[i + 1][j] * output_h_right[i][j].t();
                tmp2 += deltad2_cell_left[i + 1][j] * pow(output_h_right[i][j].t(), 2.0);
            }
        }
        hLayers[i].Ugrad_cell_left = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].U_cell_left;
        hLayers[i].Ud2_cell_left = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_output_left[i + 1][0] * output_h_left[i][0].t();
        tmp2 = deltad2_output_left[i + 1][0] * pow(output_h_left[i][0].t(), 2.0);
        if(i > 0){
            tmp += delta_output_left[i + 1][0] * output_h_right[i][0].t();
            tmp2 += deltad2_output_left[i + 1][0] * pow(output_h_right[i][0].t(), 2.0);
        }
        for(int j = 1; j < T; ++j){
            tmp += delta_output_left[i + 1][j] * output_h_left[i][j].t();
            tmp2 += deltad2_output_left[i + 1][j] * pow(output_h_left[i][j].t(), 2.0);
            if(i > 0){
                tmp += delta_output_left[i + 1][j] * output_h_right[i][j].t();
                tmp2 += deltad2_output_left[i + 1][j] * pow(output_h_right[i][j].t(), 2.0);
            }
        }
        hLayers[i].Ugrad_output_left = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].U_output_left;
        hLayers[i].Ud2_output_left = tmp2 / nSamples + hiddenConfig[i].WeightDecay;
        // W
        tmp = delta_input_left[i + 1][T - 1] * output_h_left[i + 1][T - 2].t();
        tmp2 = deltad2_input_left[i + 1][T - 1] * pow(output_h_left[i + 1][T - 2].t(), 2.0);

        for(int j = T - 2; j > 0; j--){
            tmp += delta_input_left[i + 1][j] * output_h_left[i + 1][j - 1].t();
            tmp2 += deltad2_input_left[i + 1][j] * pow(output_h_left[i + 1][j - 1].t(), 2.0);
        }
        hLayers[i].Wgrad_input_left = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].W_input_left;
        hLayers[i].Wd2_input_left = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_forget_left[i + 1][T - 1] * output_h_left[i + 1][T - 2].t();
        tmp2 = deltad2_forget_left[i + 1][T - 1] * pow(output_h_left[i + 1][T - 2].t(), 2.0);
        for(int j = T - 2; j > 0; j--){
            tmp += delta_forget_left[i + 1][j] * output_h_left[i + 1][j - 1].t();
            tmp2 += deltad2_forget_left[i + 1][j] * pow(output_h_left[i + 1][j - 1].t(), 2.0);
        }
        hLayers[i].Wgrad_forget_left = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].W_forget_left;
        hLayers[i].Wd2_forget_left = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_cell_left[i + 1][T - 1] * output_h_left[i + 1][T - 2].t();
        tmp2 = deltad2_cell_left[i + 1][T - 1] * pow(output_h_left[i + 1][T - 2].t(), 2.0);
        for(int j = T - 2; j > 0; j--){
            tmp += delta_cell_left[i + 1][j] * output_h_left[i + 1][j - 1].t();
            tmp2 += deltad2_cell_left[i + 1][j] * pow(output_h_left[i + 1][j - 1].t(), 2.0);
        }
        hLayers[i].Wgrad_cell_left = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].W_cell_left;
        hLayers[i].Wd2_cell_left = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_output_left[i + 1][T - 1] * output_h_left[i + 1][T - 2].t();
        tmp2 = deltad2_output_left[i + 1][T - 1] * pow(output_h_left[i + 1][T - 2].t(), 2.0);
        for(int j = T - 2; j > 0; j--){
            tmp += delta_output_left[i + 1][j] * output_h_left[i + 1][j - 1].t();
            tmp2 += deltad2_output_left[i + 1][j] * pow(output_h_left[i + 1][j - 1].t(), 2.0);
        }
        hLayers[i].Wgrad_output_left = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].W_output_left;
        hLayers[i].Wd2_output_left = tmp2 / nSamples + hiddenConfig[i].WeightDecay;
        // V
        tmp = delta_input_left[i + 1][T - 1] * acti_cell_left[i][T - 2].t();
        tmp2 = deltad2_input_left[i + 1][T - 1] * pow(acti_cell_left[i][T - 2].t(), 2.0);
        for(int j = T - 2; j > 0; j--){
            tmp += delta_input_left[i + 1][j] * acti_cell_left[i][j - 1].t();
            tmp2 += deltad2_input_left[i + 1][j] * pow(acti_cell_left[i][j - 1].t(), 2.0);
        }
        hLayers[i].Vgrad_input_left -> full = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].V_input_left -> full;
        hLayers[i].Vgrad_input_left -> update(UPDATE_FROM_FULL);
        hLayers[i].Vgrad_input_left -> update(UPDATE_FROM_DIAG);
        hLayers[i].Vd2_input_left -> full = tmp2 / nSamples + hiddenConfig[i].WeightDecay;
        hLayers[i].Vd2_input_left -> update(UPDATE_FROM_FULL);
        hLayers[i].Vd2_input_left -> update(UPDATE_FROM_DIAG);

        tmp = delta_forget_left[i + 1][T - 1] * acti_cell_left[i][T - 2].t();
        tmp2 = deltad2_forget_left[i + 1][T - 1] * pow(acti_cell_left[i][T - 2].t(), 2.0);
        for(int j = T - 2; j > 0; j--){
            tmp += delta_forget_left[i + 1][j] * acti_cell_left[i][j - 1].t();
            tmp2 += deltad2_forget_left[i + 1][j] * pow(acti_cell_left[i][j - 1].t(), 2.0);
        }
        hLayers[i].Vgrad_forget_left -> full = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].V_forget_left -> full;
        hLayers[i].Vgrad_forget_left -> update(UPDATE_FROM_FULL);
        hLayers[i].Vgrad_forget_left -> update(UPDATE_FROM_DIAG);
        hLayers[i].Vd2_forget_left -> full = tmp2 / nSamples + hiddenConfig[i].WeightDecay;
        hLayers[i].Vd2_forget_left -> update(UPDATE_FROM_FULL);
        hLayers[i].Vd2_forget_left -> update(UPDATE_FROM_DIAG);

        tmp = delta_output_left[i + 1][T - 1] * acti_cell_left[i][T - 2].t();
        tmp2 = deltad2_output_left[i + 1][T - 1] * pow(acti_cell_left[i][T - 2].t(), 2.0);
        for(int j = T - 2; j > 0; j--){
            tmp += delta_output_left[i + 1][j] * acti_cell_left[i][j - 1].t();
            tmp2 += deltad2_output_left[i + 1][j] * pow(acti_cell_left[i][j - 1].t(), 2.0);
        }
        hLayers[i].Vgrad_output_left -> full = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].V_output_left -> full;
        hLayers[i].Vgrad_output_left -> update(UPDATE_FROM_FULL);
        hLayers[i].Vgrad_output_left -> update(UPDATE_FROM_DIAG);
        hLayers[i].Vd2_output_left -> full = tmp2 / nSamples + hiddenConfig[i].WeightDecay;
        hLayers[i].Vd2_output_left -> update(UPDATE_FROM_FULL);
        hLayers[i].Vd2_output_left -> update(UPDATE_FROM_DIAG);
        // backward part
        // U
        tmp = delta_input_right[i + 1][0] * output_h_right[i][0].t();
        tmp2 = deltad2_input_right[i + 1][0] * pow(output_h_right[i][0].t(), 2.0);
        if(i > 0){
            tmp += delta_input_right[i + 1][0] * output_h_left[i][0].t();
            tmp2 += deltad2_input_right[i + 1][0] * pow(output_h_left[i][0].t(), 2.0);
        }
        for(int j = 1; j < T; ++j){
            tmp += delta_input_right[i + 1][j] * output_h_right[i][j].t();
            tmp2 += deltad2_input_right[i + 1][j] * pow(output_h_right[i][j].t(), 2.0);
            if(i > 0){
                tmp += delta_input_right[i + 1][j] * output_h_left[i][j].t();
                tmp2 += deltad2_input_right[i + 1][j] * pow(output_h_left[i][j].t(), 2.0);
            }
        }
        hLayers[i].Ugrad_input_right = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].U_input_right;
        hLayers[i].Ud2_input_right = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_forget_right[i + 1][0] * output_h_right[i][0].t();
        tmp2 = deltad2_forget_right[i + 1][0] * pow(output_h_right[i][0].t(), 2.0);
        if(i > 0){
            tmp += delta_forget_right[i + 1][0] * output_h_left[i][0].t();
            tmp2 += deltad2_forget_right[i + 1][0] * pow(output_h_left[i][0].t(), 2.0);
        }
        for(int j = 1; j < T; ++j){
            tmp += delta_forget_right[i + 1][j] * output_h_right[i][j].t();
            tmp2 += deltad2_forget_right[i + 1][j] * pow(output_h_right[i][j].t(), 2.0);
            if(i > 0){
                tmp += delta_forget_right[i + 1][j] * output_h_left[i][j].t();
                tmp2 += deltad2_forget_right[i + 1][j] * pow(output_h_left[i][j].t(), 2.0);
            }
        }
        hLayers[i].Ugrad_forget_right = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].U_forget_right;
        hLayers[i].Ud2_forget_right = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_cell_right[i + 1][0] * output_h_right[i][0].t();
        tmp2 = deltad2_cell_right[i + 1][0] * pow(output_h_right[i][0].t(), 2.0);
        if(i > 0){
            tmp += delta_cell_right[i + 1][0] * output_h_left[i][0].t();
            tmp2 += deltad2_cell_right[i + 1][0] * pow(output_h_left[i][0].t(), 2.0);
        }
        for(int j = 1; j < T; ++j){
            tmp += delta_cell_right[i + 1][j] * output_h_right[i][j].t();
            tmp2 += deltad2_cell_right[i + 1][j] * pow(output_h_right[i][j].t(), 2.0);
            if(i > 0){
                tmp += delta_cell_right[i + 1][j] * output_h_left[i][j].t();
                tmp2 += deltad2_cell_right[i + 1][j] * pow(output_h_left[i][j].t(), 2.0);
            }
        }
        hLayers[i].Ugrad_cell_right = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].U_cell_right;
        hLayers[i].Ud2_cell_right = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_output_right[i + 1][0] * output_h_right[i][0].t();
        tmp2 = deltad2_output_right[i + 1][0] * pow(output_h_right[i][0].t(), 2.0);
        if(i > 0){
            tmp += delta_output_right[i + 1][0] * output_h_left[i][0].t();
            tmp2 += deltad2_output_right[i + 1][0] * pow(output_h_left[i][0].t(), 2.0);
        }
        for(int j = 1; j < T; ++j){
            tmp += delta_output_right[i + 1][j] * output_h_right[i][j].t();
            tmp2 += deltad2_output_right[i + 1][j] * pow(output_h_right[i][j].t(), 2.0);
            if(i > 0){
                tmp += delta_output_right[i + 1][j] * output_h_left[i][j].t();
                tmp2 += deltad2_output_right[i + 1][j] * pow(output_h_left[i][j].t(), 2.0);
            }
        }
        hLayers[i].Ugrad_output_right = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].U_output_right;
        hLayers[i].Ud2_output_right = tmp2 / nSamples + hiddenConfig[i].WeightDecay;
        // W
        tmp = delta_input_right[i + 1][0] * output_h_right[i + 1][1].t();
        tmp2 = deltad2_input_right[i + 1][0] * pow(output_h_right[i + 1][1].t(), 2.0);
        for(int j = 1; j < T - 1; ++j){
            tmp += delta_input_right[i + 1][j] * output_h_right[i + 1][j + 1].t();
            tmp2 += deltad2_input_right[i + 1][j] * pow(output_h_right[i + 1][j + 1].t(), 2.0);
        }
        hLayers[i].Wgrad_input_right = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].W_input_right;
        hLayers[i].Wd2_input_right = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_forget_right[i + 1][0] * output_h_right[i + 1][1].t();
        tmp2 = deltad2_forget_right[i + 1][0] * pow(output_h_right[i + 1][1].t(), 2.0);
        for(int j = 1; j < T - 1; ++j){
            tmp += delta_forget_right[i + 1][j] * output_h_right[i + 1][j + 1].t();
            tmp2 += deltad2_forget_right[i + 1][j] * pow(output_h_right[i + 1][j + 1].t(), 2.0);
        }
        hLayers[i].Wgrad_forget_right = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].W_forget_right;
        hLayers[i].Wd2_forget_right = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_cell_right[i + 1][0] * output_h_right[i + 1][1].t();
        tmp2 = deltad2_cell_right[i + 1][0] * pow(output_h_right[i + 1][1].t(), 2.0);
        for(int j = 1; j < T - 1; ++j){
            tmp += delta_cell_right[i + 1][j] * output_h_right[i + 1][j + 1].t();
            tmp2 += deltad2_cell_right[i + 1][j] * pow(output_h_right[i + 1][j + 1].t(), 2.0);
        }
        hLayers[i].Wgrad_cell_right = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].W_cell_right;
        hLayers[i].Wd2_cell_right = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_output_right[i + 1][0] * output_h_right[i + 1][1].t();
        tmp2 = deltad2_output_right[i + 1][0] * pow(output_h_right[i + 1][1].t(), 2.0);
        for(int j = 1; j < T - 1; ++j){
            tmp += delta_output_right[i + 1][j] * output_h_right[i + 1][j + 1].t();
            tmp2 += deltad2_output_right[i + 1][j] * pow(output_h_right[i + 1][j + 1].t(), 2.0);
        }
        hLayers[i].Wgrad_output_right = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].W_output_right;
        hLayers[i].Wd2_output_right = tmp2 / nSamples + hiddenConfig[i].WeightDecay;
        // V
        tmp = delta_input_right[i + 1][0] * acti_cell_right[i][1].t();
        tmp2 = deltad2_input_right[i + 1][0] * pow(acti_cell_right[i][1].t(), 2.0);
        for(int j = 1; j < T - 1; ++j){
            tmp += delta_input_right[i + 1][j] * acti_cell_right[i][j + 1].t();
            tmp2 += deltad2_input_right[i + 1][j] * pow(acti_cell_right[i][j + 1].t(), 2.0);
        }
        hLayers[i].Vgrad_input_right -> full = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].V_input_right -> full;
        hLayers[i].Vgrad_input_right -> update(UPDATE_FROM_FULL);
        hLayers[i].Vgrad_input_right -> update(UPDATE_FROM_DIAG);
        hLayers[i].Vd2_input_right -> full = tmp2 / nSamples + hiddenConfig[i].WeightDecay;
        hLayers[i].Vd2_input_right -> update(UPDATE_FROM_FULL);
        hLayers[i].Vd2_input_right -> update(UPDATE_FROM_DIAG);

        tmp = delta_forget_right[i + 1][0] * acti_cell_right[i][1].t();
        tmp2 = deltad2_forget_right[i + 1][0] * pow(acti_cell_right[i][1].t(), 2.0);
        for(int j = 1; j < T - 1; ++j){
            tmp += delta_forget_right[i + 1][j] * acti_cell_right[i][j + 1].t();
            tmp2 += deltad2_forget_right[i + 1][j] * pow(acti_cell_right[i][j + 1].t(), 2.0);
        }
        hLayers[i].Vgrad_forget_right -> full = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].V_forget_right -> full;
        hLayers[i].Vgrad_forget_right -> update(UPDATE_FROM_FULL);
        hLayers[i].Vgrad_forget_right -> update(UPDATE_FROM_DIAG);
        hLayers[i].Vd2_forget_right -> full = tmp2 / nSamples + hiddenConfig[i].WeightDecay;
        hLayers[i].Vd2_forget_right -> update(UPDATE_FROM_FULL);
        hLayers[i].Vd2_forget_right -> update(UPDATE_FROM_DIAG);

        tmp = delta_output_right[i + 1][0] * acti_cell_right[i][1].t();
        tmp2 = deltad2_output_right[i + 1][0] * pow(acti_cell_right[i][1].t(), 2.0);
        for(int j = 1; j < T - 1; ++j){
            tmp += delta_output_right[i + 1][j] * acti_cell_right[i][j + 1].t();
            tmp2 += deltad2_output_right[i + 1][j] * pow(acti_cell_right[i][j + 1].t(), 2.0);
        }
        hLayers[i].Vgrad_output_right -> full = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].V_output_right -> full;
        hLayers[i].Vgrad_output_right -> update(UPDATE_FROM_FULL);
        hLayers[i].Vgrad_output_right -> update(UPDATE_FROM_DIAG);
        hLayers[i].Vd2_output_right -> full = tmp2 / nSamples + hiddenConfig[i].WeightDecay;
        hLayers[i].Vd2_output_right -> update(UPDATE_FROM_FULL);
        hLayers[i].Vd2_output_right -> update(UPDATE_FROM_DIAG);
        //*/
    }

    // destructor
    nonlin_input_left.clear();
    std::vector<std::vector<Mat> > ().swap(nonlin_input_left);
    nonlin_forget_left.clear();
    std::vector<std::vector<Mat> > ().swap(nonlin_forget_left);
    nonlin_cell_left.clear();
    std::vector<std::vector<Mat> > ().swap(nonlin_cell_left);
    nonlin_output_left.clear();
    std::vector<std::vector<Mat> > ().swap(nonlin_output_left);
    acti_input_left.clear();
    std::vector<std::vector<Mat> > ().swap(acti_input_left);
    acti_forget_left.clear();
    std::vector<std::vector<Mat> > ().swap(acti_forget_left);
    acti_cell_left.clear();
    std::vector<std::vector<Mat> > ().swap(acti_cell_left);
    acti_output_left.clear();
    std::vector<std::vector<Mat> > ().swap(acti_output_left);
    output_h_left.clear();
    std::vector<std::vector<Mat> > ().swap(output_h_left);

    delta_input_left.clear();
    std::vector<std::vector<Mat> > ().swap(delta_input_left);
    deltad2_input_left.clear();
    std::vector<std::vector<Mat> > ().swap(deltad2_input_left);
    delta_forget_left.clear();
    std::vector<std::vector<Mat> > ().swap(delta_forget_left);
    deltad2_forget_left.clear();
    std::vector<std::vector<Mat> > ().swap(deltad2_forget_left);
    delta_cell_left.clear();
    std::vector<std::vector<Mat> > ().swap(delta_cell_left);
    deltad2_cell_left.clear();
    std::vector<std::vector<Mat> > ().swap(deltad2_cell_left);
    delta_output_left.clear();
    std::vector<std::vector<Mat> > ().swap(delta_output_left);
    deltad2_output_left.clear();
    std::vector<std::vector<Mat> > ().swap(deltad2_output_left);
    epsilon_output_left.clear();
    std::vector<std::vector<Mat> > ().swap(epsilon_output_left);
    epsilond2_output_left.clear();
    std::vector<std::vector<Mat> > ().swap(epsilond2_output_left);
    epsilon_state_left.clear();
    std::vector<std::vector<Mat> > ().swap(epsilon_state_left);
    epsilond2_state_left.clear();
    std::vector<std::vector<Mat> > ().swap(epsilond2_state_left);

    nonlin_input_right.clear();
    std::vector<std::vector<Mat> > ().swap(nonlin_input_right);
    nonlin_forget_right.clear();
    std::vector<std::vector<Mat> > ().swap(nonlin_forget_right);
    nonlin_cell_right.clear();
    std::vector<std::vector<Mat> > ().swap(nonlin_cell_right);
    nonlin_output_right.clear();
    std::vector<std::vector<Mat> > ().swap(nonlin_output_right);
    acti_input_right.clear();
    std::vector<std::vector<Mat> > ().swap(acti_input_right);
    acti_forget_right.clear();
    std::vector<std::vector<Mat> > ().swap(acti_forget_right);
    acti_cell_right.clear();
    std::vector<std::vector<Mat> > ().swap(acti_cell_right);
    acti_output_right.clear();
    std::vector<std::vector<Mat> > ().swap(acti_output_right);
    output_h_right.clear();
    std::vector<std::vector<Mat> > ().swap(output_h_right);

    delta_input_right.clear();
    std::vector<std::vector<Mat> > ().swap(delta_input_right);
    deltad2_input_right.clear();
    std::vector<std::vector<Mat> > ().swap(deltad2_input_right);
    delta_forget_right.clear();
    std::vector<std::vector<Mat> > ().swap(delta_forget_right);
    deltad2_forget_right.clear();
    std::vector<std::vector<Mat> > ().swap(deltad2_forget_right);
    delta_cell_right.clear();
    std::vector<std::vector<Mat> > ().swap(delta_cell_right);
    deltad2_cell_right.clear();
    std::vector<std::vector<Mat> > ().swap(deltad2_cell_right);
    delta_output_right.clear();
    std::vector<std::vector<Mat> > ().swap(delta_output_right);
    deltad2_output_right.clear();
    std::vector<std::vector<Mat> > ().swap(deltad2_output_right);
    epsilon_output_right.clear();
    std::vector<std::vector<Mat> > ().swap(epsilon_output_right);
    epsilond2_output_right.clear();
    std::vector<std::vector<Mat> > ().swap(epsilond2_output_right);
    epsilon_state_right.clear();
    std::vector<std::vector<Mat> > ().swap(epsilon_state_right);
    epsilond2_state_right.clear();
    std::vector<std::vector<Mat> > ().swap(epsilond2_state_right);

    groundTruth.clear();
    std::vector<Mat>().swap(groundTruth);
    p.clear();
    std::vector<Mat>().swap(p);
    return true;

}

