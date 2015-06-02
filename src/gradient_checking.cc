#include "gradient_checking.h"

using namespace cv;
using namespace std;


void
gradientChecking_SoftmaxLayer(std::vector<LSTMl> &hLayers, Smr &smr, std::vector<Mat> &sampleX, Mat &sampleY){
    //Gradient Checking (remember to disable this part after you're sure the 
    //cost function and dJ function are correct)
    getNetworkCost(sampleX, sampleY, hLayers, smr);

    Mat grad;
    smr.Wgrad.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test softmax layer !!!!"<<endl;
    cout<<"################################################"<<endl;
    double epsilon = 1e-4;
    for(int i = 0; i < smr.W.rows; i++){
        for(int j = 0; j < smr.W.cols; j++){
            double memo = smr.W.ATD(i, j);
            smr.W.ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            smr.W.ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            smr.W.ATD(i, j) = memo;
        }
    }
    grad.release();
    
}

void
gradientChecking_RecurrentLayer(std::vector<Rl> &hLayers, Smr &smr, std::vector<Mat> &sampleX, Mat &sampleY, int layer){
    //Gradient Checking (remember to disable this part after you're sure the 
    //cost function and dJ function are correct)
    getNetworkCost(sampleX, sampleY, hLayers, smr);
    Mat grad;
    double epsilon = 1e-4;
    
    hLayers[layer].Wgrad.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test Recurrent layer["<<layer<<"] W !!!!"<<endl;
    cout<<"################################################"<<endl;
    for(int i = 0; i < hLayers[layer].W.rows; i++){
        for(int j = 0; j < hLayers[layer].W.cols; j++){
            double memo = hLayers[layer].W.ATD(i, j);
            hLayers[layer].W.ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            hLayers[layer].W.ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) ;//cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            hLayers[layer].W.ATD(i, j) = memo;
        }
    }
    hLayers[layer].Ugrad.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test Recurrent layer["<<layer<<"] U !!!!"<<endl;
    cout<<"################################################"<<endl;
    for(int i = 0; i < hLayers[layer].U.rows; i++){
        for(int j = 0; j < hLayers[layer].U.cols; j++){
            double memo = hLayers[layer].U.ATD(i, j);
            hLayers[layer].U.ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            hLayers[layer].U.ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) ;//cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            hLayers[layer].U.ATD(i, j) = memo;
        }
    }
    grad.release();
}

void
gradientChecking_LSTMLayer(std::vector<LSTMl> &hLayers, Smr &smr, std::vector<Mat> &sampleX, Mat &sampleY, int layer){
    //Gradient Checking (remember to disable this part after you're sure the 
    //cost function and dJ function are correct)
    getNetworkCost(sampleX, sampleY, hLayers, smr);
    Mat grad;
    double epsilon = 1e-4;
    //cout.precision(18);
    /*
    hLayers[layer].Wgrad_output.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test LSTM layer["<<layer<<"] W !!!!"<<endl;
    cout<<"################################################"<<endl;
    for(int i = 0; i < hLayers[layer].W_output.rows; i++){
        for(int j = 0; j < hLayers[layer].W_output.cols; j++){
            double memo = hLayers[layer].W_output.ATD(i, j);
            hLayers[layer].W_output.ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            hLayers[layer].W_output.ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            //cout<<"value1 = "<<value1<<", value2 = "<<value2<<endl;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) ;//cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            hLayers[layer].W_output.ATD(i, j) = memo;
        }
    }
    
//*/
/*
    hLayers[layer].Ugrad_output.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test LSTM layer["<<layer<<"] U !!!!"<<endl;
    cout<<"################################################"<<endl;
    for(int i = 0; i < hLayers[layer].U_output.rows; i++){
        for(int j = 0; j < hLayers[layer].U_output.cols; j++){
            double memo = hLayers[layer].U_output.ATD(i, j);
            hLayers[layer].U_output.ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            hLayers[layer].U_output.ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            //cout<<"value1 = "<<value1<<", value2 = "<<value2<<endl;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) ;//cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            hLayers[layer].U_output.ATD(i, j) = memo;
        }
    }
    //*/
   /*
    hLayers[layer].Wgrad_input.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test LSTM layer["<<layer<<"] W !!!!"<<endl;
    cout<<"################################################"<<endl;
    cout<<hLayers[layer].W_input.size()<<endl;
    for(int i = 0; i < hLayers[layer].W_input.rows; i++){
        for(int j = 0; j < hLayers[layer].W_input.cols; j++){
            double memo = hLayers[layer].W_input.ATD(i, j);
            hLayers[layer].W_input.ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            hLayers[layer].W_input.ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) ;//cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            hLayers[layer].W_input.ATD(i, j) = memo;
        }
    }
   
    hLayers[layer].Ugrad_input.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test LSTM layer["<<layer<<"] U !!!!"<<endl;
    cout<<"################################################"<<endl;
    for(int i = 0; i < hLayers[layer].U_input.rows; i++){
        for(int j = 0; j < hLayers[layer].U_input.cols; j++){
            double memo = hLayers[layer].U_input.ATD(i, j);
            hLayers[layer].U_input.ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            hLayers[layer].U_input.ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            //cout<<"value1 = "<<value1<<", value2 = "<<value2<<endl;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) ;//cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            hLayers[layer].U_input.ATD(i, j) = memo;
        }
    }
  //*/
   /*
    hLayers[layer].Wgrad_cell.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test LSTM layer["<<layer<<"] W !!!!"<<endl;
    cout<<"################################################"<<endl;
    cout<<hLayers[layer].W_cell.size()<<endl;
    for(int i = 0; i < hLayers[layer].W_cell.rows; i++){
        for(int j = 0; j < hLayers[layer].W_cell.cols; j++){
            double memo = hLayers[layer].W_cell.ATD(i, j);
            hLayers[layer].W_cell.ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            hLayers[layer].W_cell.ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) ;//cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            hLayers[layer].W_cell.ATD(i, j) = memo;
        }
    }
   
    hLayers[layer].Ugrad_cell.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test LSTM layer["<<layer<<"] U !!!!"<<endl;
    cout<<"################################################"<<endl;
    for(int i = 0; i < hLayers[layer].U_cell.rows; i++){
        for(int j = 0; j < hLayers[layer].U_cell.cols; j++){
            double memo = hLayers[layer].U_cell.ATD(i, j);
            hLayers[layer].U_cell.ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            hLayers[layer].U_cell.ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            //cout<<"value1 = "<<value1<<", value2 = "<<value2<<endl;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) ;//cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            hLayers[layer].U_cell.ATD(i, j) = memo;
        }
    }
  //*/
  //
  
/*
    hLayers[layer].Wgrad_forget.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test LSTM layer["<<layer<<"] W !!!!"<<endl;
    cout<<"################################################"<<endl;
    for(int i = 0; i < hLayers[layer].W_forget.rows; i++){
        for(int j = 0; j < hLayers[layer].W_forget.cols; j++){
            double memo = hLayers[layer].W_forget.ATD(i, j);
            hLayers[layer].W_forget.ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            hLayers[layer].W_forget.ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) ;//cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            hLayers[layer].W_forget.ATD(i, j) = memo;
        }
    }
  
    hLayers[layer].Ugrad_forget.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test LSTM layer["<<layer<<"] U !!!!"<<endl;
    cout<<"################################################"<<endl;
    for(int i = 0; i < hLayers[layer].U_forget.rows; i++){
        for(int j = 0; j < hLayers[layer].U_forget.cols; j++){
            double memo = hLayers[layer].U_forget.ATD(i, j);
            hLayers[layer].U_forget.ATD(i, j) = memo + epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value1 = smr.cost;
            hLayers[layer].U_forget.ATD(i, j) = memo - epsilon;
            getNetworkCost(sampleX, sampleY, hLayers, smr);
            double value2 = smr.cost;
            //cout<<"value1 = "<<value1<<", value2 = "<<value2<<endl;
            double tp = (value1 - value2) / (2 * epsilon);
            if(tp == 0.0 && grad.ATD(i, j) == 0.0) ;//cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
            else cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<tp / grad.ATD(i, j)<<endl;
            hLayers[layer].U_forget.ATD(i, j) = memo;
        }
    }
  //*/


/*
    hLayers[layer].Vgrad_input -> diagonal.copyTo(grad);
    cout<<"################################################"<<endl;
    cout<<"## test LSTM layer["<<layer<<"] W !!!!"<<endl;
    cout<<"################################################"<<endl;
    for(int i = 0; i < hLayers[layer].V_input -> diagonal.rows; i++){
        double memo = hLayers[layer].V_input -> diagonal.ATD(i, 0);
        hLayers[layer].V_input -> diagonal.ATD(i, 0) = memo + epsilon;
        getNetworkCost(sampleX, sampleY, hLayers, smr);
        double value1 = smr.cost;
        hLayers[layer].V_input -> diagonal.ATD(i, 0) = memo - epsilon;
        getNetworkCost(sampleX, sampleY, hLayers, smr);
        double value2 = smr.cost;
        double tp = (value1 - value2) / (2 * epsilon);
        if(tp == 0.0 && grad.ATD(i, 0) == 0.0) ;//cout<<i<<", "<<j<<", "<<tp<<", "<<grad.ATD(i, j)<<", "<<1<<endl;
        else cout<<i<<", "<<0<<", "<<tp<<", "<<grad.ATD(i, 0)<<", "<<tp / grad.ATD(i, 0)<<endl;
        hLayers[layer].V_input -> diagonal.ATD(i, 0) = memo;
    }   
*/
    grad.release();
}

