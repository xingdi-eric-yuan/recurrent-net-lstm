#include "result_predict.h"

using namespace cv;
using namespace std;

Mat 
resultPredict(std::vector<Mat> &x, std::vector<LSTMl> &hLayers, Smr &smr){

    int T = x.size();
    // hidden layer forward
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
            tmp_input = Tanh(tmp_input);
            tmp_forget = Tanh(tmp_forget);
            tmp_cell = sigmoid(tmp_cell);
            tmp_input.copyTo(acti_input[i - 1][j]);
            tmp_forget.copyTo(acti_forget[i - 1][j]);
            tmp_cell = tmp_cell.mul(tmp_input);
            if(j > 0){
                tmp_cell += tmp_forget.mul(acti_cell[i - 1][j - 1]);
            }
            tmp_cell.copyTo(acti_cell[i - 1][j]);
            tmp_output += hLayers[i - 1].V_output -> full * tmp_cell;
            tmp_output = Tanh(tmp_output);
            tmp_output.copyTo(acti_output[i - 1][j]);
            tmp_output = tmp_output.mul(sigmoid(tmp_cell));
            tmp_output.copyTo(output_h[i][j]);
        }
    }
    // softmax layer forward
    Mat M = smr.W * output_h[output_h.size() - 1][T - 1];
    Mat result = Mat::zeros(1, M.cols, CV_64FC1);

    double minValue, maxValue;
    Point minLoc, maxLoc;
    for(int i = 0; i < M.cols; i++){
        minMaxLoc(M(Rect(i, 0, 1, M.rows)), &minValue, &maxValue, &minLoc, &maxLoc);
        result.ATD(0, i) = (int)maxLoc.y;
    }
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
    return result;
}

Mat 
resultPredict(std::vector<Mat> &x, std::vector<Rl> &hLayers, Smr &smr){

    int T = x.size();
    // hidden layer forward
    std::vector<std::vector<Mat> > acti;
    std::vector<Mat> tmp_vec;
    acti.push_back(tmp_vec);
    for(int i = 0; i < T; ++i){
        acti[0].push_back(x[i]);
    }
    for(int i = 1; i <= hiddenConfig.size(); ++i){
        // for each hidden layer
        acti.push_back(tmp_vec);
        for(int j = 0; j < T; ++j){
            // for each time slot
            Mat tmpacti = hLayers[i - 1].U * acti[i - 1][j];
            if(j > 0) tmpacti += hLayers[i - 1].W * acti[i][j - 1];
            tmpacti = ReLU(tmpacti);
            if(hiddenConfig[i - 1].DropoutRate < 1.0) tmpacti = tmpacti.mul(hiddenConfig[i - 1].DropoutRate);
            acti[i].push_back(tmpacti);
        }
    }

    // softmax layer forward
    Mat M = smr.W * acti[acti.size() - 1][T - 1];
    Mat result = Mat::zeros(1, M.cols, CV_64FC1);

    double minValue, maxValue;
    Point minLoc, maxLoc;
    for(int i = 0; i < M.cols; i++){
        minMaxLoc(M(Rect(i, 0, 1, M.rows)), &minValue, &maxValue, &minLoc, &maxLoc);
        result.ATD(0, i) = (int)maxLoc.y;
    }
    acti.clear();
    std::vector<std::vector<Mat> >().swap(acti);
    return result;
}

void 
testNetwork(const std::vector<std::vector<int> > &x, std::vector<std::vector<int> > &y, std::vector<LSTMl> &HiddenLayers, Smr &smr, 
             std::vector<string> &re_wordmap){

    // Test use test set
    // Because it may leads to lack of memory if testing the whole dataset at 
    // one time, so separate the dataset into small pieces of batches (say, batch size = 20).
    // 
    int batchSize = 50;
    Mat result = Mat::zeros(1, x.size(), CV_64FC1);

    std::vector<std::vector<int> > tmpBatch;
    int batch_amount = x.size() / batchSize;
    for(int i = 0; i < batch_amount; i++){
        for(int j = 0; j < batchSize; j++){
            tmpBatch.push_back(x[i * batchSize + j]);
        }
        std::vector<Mat> sampleX;
        getDataMat(tmpBatch, sampleX, re_wordmap);
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
        getDataMat(tmpBatch, sampleX, re_wordmap);
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


void 
testNetwork(const std::vector<std::vector<int> > &x, std::vector<std::vector<int> > &y, std::vector<Rl> &HiddenLayers, Smr &smr, 
             std::vector<string> &re_wordmap){

    // Test use test set
    // Because it may leads to lack of memory if testing the whole dataset at 
    // one time, so separate the dataset into small pieces of batches (say, batch size = 20).
    // 
    int batchSize = 50;
    Mat result = Mat::zeros(1, x.size(), CV_64FC1);

    std::vector<std::vector<int> > tmpBatch;
    int batch_amount = x.size() / batchSize;
    for(int i = 0; i < batch_amount; i++){
        for(int j = 0; j < batchSize; j++){
            tmpBatch.push_back(x[i * batchSize + j]);
        }
        std::vector<Mat> sampleX;
        getDataMat(tmpBatch, sampleX, re_wordmap);
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
        getDataMat(tmpBatch, sampleX, re_wordmap);
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



