#include "result_predict.h"

using namespace cv;
using namespace std;

Mat 
resultPredict(std::vector<Mat> &x, std::vector<LSTMl> &hLayers, Smr &smr){

    int T = x.size();
    int mid = (int)(T /2.0);
    // hidden layer forward
    std::vector<std::vector<Mat> > acti_input_left;
    std::vector<std::vector<Mat> > acti_forget_left;
    std::vector<std::vector<Mat> > acti_cell_left;
    std::vector<std::vector<Mat> > acti_output_left;
    std::vector<std::vector<Mat> > output_h_left;

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
        acti_input_left.push_back(tmp_vec);
        acti_forget_left.push_back(tmp_vec);
        acti_cell_left.push_back(tmp_vec);
        acti_output_left.push_back(tmp_vec);
        output_h_left.push_back(tmp_vec);

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
                tmp_input = hLayers[i - 1].U_input_left * output_h_right[i - 1][j];
                tmp_forget = hLayers[i - 1].U_forget_left * output_h_right[i - 1][j];
                tmp_cell = hLayers[i - 1].U_cell_left * output_h_right[i - 1][j];
                tmp_output = hLayers[i - 1].U_output_left * output_h_right[i - 1][j];
            } 
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
                tmp_input = hLayers[i - 1].U_input_right * output_h_left[i - 1][j];
                tmp_forget = hLayers[i - 1].U_forget_right * output_h_left[i - 1][j];
                tmp_cell = hLayers[i - 1].U_cell_right * output_h_left[i - 1][j];
                tmp_output = hLayers[i - 1].U_output_right * output_h_left[i - 1][j];
            }
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
            tmp_output = nonLinearity(tmp_output, GATE_NL);
            tmp_output.copyTo(acti_output_right[i - 1][j]);
            tmp_output = tmp_output.mul(nonLinearity(tmp_cell, IO_NL));
            tmp_output.copyTo(output_h_right[i][j]);
        }
    }

    // softmax layer forward
    Mat M = smr.W_left * output_h_left[output_h_left.size() - 1][mid];
    M += smr.W_right * output_h_right[output_h_right.size() - 1][mid];
    Mat result = Mat::zeros(1, M.cols, CV_64FC1);

    double minValue, maxValue;
    Point minLoc, maxLoc;
    for(int i = 0; i < M.cols; i++){
        minMaxLoc(M(Rect(i, 0, 1, M.rows)), &minValue, &maxValue, &minLoc, &maxLoc);
        result.ATD(0, i) = (int)maxLoc.y;
    }

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
    return result;
}
void 
testNetwork(const std::vector<std::vector<int> > &x, std::vector<std::vector<int> > &y, std::vector<LSTMl> &HiddenLayers, Smr &smr, 
             std::vector<string> &re_wordmap, std::unordered_map<string, Mat> &wordvec){

    // Test use test set
    // Because it may leads to lack of memory if testing the whole dataset at 
    // one time, so separate the dataset into small pieces of batches (say, batch size = 20).
    int batchSize = 50;
    Mat result = Mat::zeros(1, x.size(), CV_64FC1);
    std::vector<std::vector<int> > tmpBatch;
    int batch_amount = x.size() / batchSize;
    for(int i = 0; i < batch_amount; i++){
        for(int j = 0; j < batchSize; j++){
            tmpBatch.push_back(x[i * batchSize + j]);
        }
        std::vector<Mat> sampleX;
        if(use_word2vec){
            getDataMat(tmpBatch, sampleX, re_wordmap, wordvec);
        }else{
            getDataMat(tmpBatch, sampleX, re_wordmap);
        }
        Mat resultBatch = resultPredict(sampleX, HiddenLayers, smr);
        Rect roi = Rect(i * batchSize, 0, batchSize, 1);
        resultBatch.copyTo(result(roi));
        tmpBatch.clear();
        sampleX.clear();
    }
    if(x.size() % batchSize){
        for(int j = 0; j < x.size() % batchSize; j++){
            tmpBatch.push_back(x[batch_amount * batchSize + j]);
        }
        std::vector<Mat> sampleX;
        if(use_word2vec){
            getDataMat(tmpBatch, sampleX, re_wordmap, wordvec);
        }else{
            getDataMat(tmpBatch, sampleX, re_wordmap);
        }
        Mat resultBatch = resultPredict(sampleX, HiddenLayers, smr);
        Rect roi = Rect(batch_amount * batchSize, 0, x.size() % batchSize, 1);
        resultBatch.copyTo(result(roi));
        ++ batch_amount;
        tmpBatch.clear();
        sampleX.clear();
    }
    Mat sampleY = Mat::zeros(1, y.size(), CV_64FC1);
    getLabelMat(y, sampleY);
    Mat err;
    sampleY.copyTo(err);
    err -= result;
    int correct = err.cols;
    for(int i=0; i<err.cols; i++){
        if(err.ATD(0, i) != 0) --correct;
    }
    cout<<"######################################"<<endl;
    cout<<"## test result. "<<correct<<" correct of "<<err.cols<<" total."<<endl;
    cout<<"## Accuracy is "<<(double)correct / (double)(err.cols)<<endl;
    cout<<"######################################"<<endl;
    result.release();
    err.release();
}

