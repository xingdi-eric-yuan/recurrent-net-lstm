#include "gradient_checking.h"

using namespace cv;
using namespace std;
void 
gradient_checking(std::vector<Mat> &sampleX, Mat &sampleY, std::vector<LSTMl> &hLayers, Smr &smr, Mat &gradient, Mat* alt){
    Mat grad;
    gradient.copyTo(grad);
    double epsilon = 1e-4;
    for(int i = 0; i < alt -> rows; i++){
        for(int j = 0; j < alt -> cols; j++){
            double memo = alt -> ATD(i, j);
            alt -> ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            alt -> ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            alt -> ATD(i, j) = memo;
        }
    }
    grad.release();
}

void
gradientChecking_SoftmaxLayer(std::vector<LSTMl> &hLayers, Smr &smr, std::vector<Mat> &sampleX, Mat &sampleY){
    //Gradient Checking (remember to disable this part after you're sure the 
    //cost function and dJ function are correct)
    getNetworkCost(sampleX, sampleY, hLayers, smr);
    Mat *p;
    cout<<"################################################"<<endl;
    cout<<"## test softmax layer !!!! --- forward"<<endl;
    cout<<"################################################"<<endl;
    p = &(smr.W_left);
    gradient_checking(sampleX, sampleY, hLayers, smr, smr.Wgrad_left, p);

    cout<<"################################################"<<endl;
    cout<<"## test softmax layer !!!! --- backward"<<endl;
    cout<<"################################################"<<endl;
    p = &(smr.W_right);
    gradient_checking(sampleX, sampleY, hLayers, smr, smr.Wgrad_right, p);    
}

void
gradientChecking_LSTMLayer(std::vector<LSTMl> &hLayers, Smr &smr, std::vector<Mat> &sampleX, Mat &sampleY, int layer){
    //Gradient Checking (remember to disable this part after you're sure the 
    //cost function and dJ function are correct)
    getNetworkCost(sampleX, sampleY, hLayers, smr);
    int which_to_check = 1;
    Mat *p;
    // which 2 check
    // 0 : all
    // 1 : output
    // 2 : cell
    // 3 : forget
    // 4 : input
    if(which_to_check == 0 || which_to_check == 1){
        cout<<"################################################"<<endl;
        cout<<"## test LSTM layer["<<layer<<"] output W --- forward"<<endl;
        cout<<"################################################"<<endl;
        p = &(hLayers[layer].W_output_left);
        gradient_checking(sampleX, sampleY, hLayers, smr, hLayers[layer].Wgrad_output_left, p);

        cout<<"################################################"<<endl;
        cout<<"## test LSTM layer["<<layer<<"] output W --- backward"<<endl;
        cout<<"################################################"<<endl;
        p = &(hLayers[layer].W_output_right);
        gradient_checking(sampleX, sampleY, hLayers, smr, hLayers[layer].Wgrad_output_right, p);

        cout<<"################################################"<<endl;
        cout<<"## test LSTM layer["<<layer<<"] output U --- forward"<<endl;
        cout<<"################################################"<<endl;
        p = &(hLayers[layer].U_output_left);
        gradient_checking(sampleX, sampleY, hLayers, smr, hLayers[layer].Ugrad_output_left, p);

        cout<<"################################################"<<endl;
        cout<<"## test LSTM layer["<<layer<<"] output U --- backward"<<endl;
        cout<<"################################################"<<endl;
        p = &(hLayers[layer].U_output_right);
        gradient_checking(sampleX, sampleY, hLayers, smr, hLayers[layer].Ugrad_output_right, p);
    }elif(which_to_check == 0 || which_to_check == 2){
        cout<<"################################################"<<endl;
        cout<<"## test LSTM layer["<<layer<<"] cell W --- forward"<<endl;
        cout<<"################################################"<<endl;
        p = &(hLayers[layer].W_cell_left);
        gradient_checking(sampleX, sampleY, hLayers, smr, hLayers[layer].Wgrad_cell_left, p);

        cout<<"################################################"<<endl;
        cout<<"## test LSTM layer["<<layer<<"] cell W --- backward"<<endl;
        cout<<"################################################"<<endl;
        p = &(hLayers[layer].W_cell_right);
        gradient_checking(sampleX, sampleY, hLayers, smr, hLayers[layer].Wgrad_cell_right, p);

        cout<<"################################################"<<endl;
        cout<<"## test LSTM layer["<<layer<<"] cell U --- forward"<<endl;
        cout<<"################################################"<<endl;
        p = &(hLayers[layer].U_cell_left);
        gradient_checking(sampleX, sampleY, hLayers, smr, hLayers[layer].Ugrad_cell_left, p);

        cout<<"################################################"<<endl;
        cout<<"## test LSTM layer["<<layer<<"] cell U --- backward"<<endl;
        cout<<"################################################"<<endl;
        p = &(hLayers[layer].U_cell_right);
        gradient_checking(sampleX, sampleY, hLayers, smr, hLayers[layer].Ugrad_cell_right, p);
    }elif(which_to_check == 0 || which_to_check == 3){
        cout<<"################################################"<<endl;
        cout<<"## test LSTM layer["<<layer<<"] forget W --- forward"<<endl;
        cout<<"################################################"<<endl;
        p = &(hLayers[layer].W_forget_left);
        gradient_checking(sampleX, sampleY, hLayers, smr, hLayers[layer].Wgrad_forget_left, p);

        cout<<"################################################"<<endl;
        cout<<"## test LSTM layer["<<layer<<"] forget W --- backward"<<endl;
        cout<<"################################################"<<endl;
        p = &(hLayers[layer].W_forget_right);
        gradient_checking(sampleX, sampleY, hLayers, smr, hLayers[layer].Wgrad_forget_right, p);

        cout<<"################################################"<<endl;
        cout<<"## test LSTM layer["<<layer<<"] forget U --- forward"<<endl;
        cout<<"################################################"<<endl;
        p = &(hLayers[layer].U_forget_left);
        gradient_checking(sampleX, sampleY, hLayers, smr, hLayers[layer].Ugrad_forget_left, p);

        cout<<"################################################"<<endl;
        cout<<"## test LSTM layer["<<layer<<"] forget U --- backward"<<endl;
        cout<<"################################################"<<endl;
        p = &(hLayers[layer].U_forget_right);
        gradient_checking(sampleX, sampleY, hLayers, smr, hLayers[layer].Ugrad_forget_right, p);

    }elif(which_to_check == 0 || which_to_check == 4){
        cout<<"################################################"<<endl;
        cout<<"## test LSTM layer["<<layer<<"] input W --- forward"<<endl;
        cout<<"################################################"<<endl;
        p = &(hLayers[layer].W_input_left);
        gradient_checking(sampleX, sampleY, hLayers, smr, hLayers[layer].Wgrad_input_left, p);

        cout<<"################################################"<<endl;
        cout<<"## test LSTM layer["<<layer<<"] input W --- backward"<<endl;
        cout<<"################################################"<<endl;
        p = &(hLayers[layer].W_input_right);
        gradient_checking(sampleX, sampleY, hLayers, smr, hLayers[layer].Wgrad_input_right, p);

        cout<<"################################################"<<endl;
        cout<<"## test LSTM layer["<<layer<<"] input U --- forward"<<endl;
        cout<<"################################################"<<endl;
        p = &(hLayers[layer].U_input_left);
        gradient_checking(sampleX, sampleY, hLayers, smr, hLayers[layer].Ugrad_input_left, p);

        cout<<"################################################"<<endl;
        cout<<"## test LSTM layer["<<layer<<"] input U --- backward"<<endl;
        cout<<"################################################"<<endl;
        p = &(hLayers[layer].U_input_right);
        gradient_checking(sampleX, sampleY, hLayers, smr, hLayers[layer].Ugrad_input_right, p);

    }else{
        ; // do nothing
    }

}

