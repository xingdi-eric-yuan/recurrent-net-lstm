#include "train_network.h"

using namespace cv;
using namespace std;

void
trainNetwork(const std::vector<std::vector<int> > &x, std::vector<std::vector<int> > &y, std::vector<LSTMl> &HiddenLayers, Smr &smr, 
             const std::vector<std::vector<int> > &tx, std::vector<std::vector<int> > &ty, std::vector<string> &re_wordmap, 
             unordered_map<string, Mat> &wordvec
             ){
    if (is_gradient_checking){
        batch_size = 2;
        std::vector<Mat> sampleX;
        Mat sampleY = Mat::zeros(nGram, batch_size, CV_64FC1);
        if(use_word2vec){
            getSample(x, sampleX, y, sampleY, re_wordmap, wordvec);
        }else{
            getSample(x, sampleX, y, sampleY, re_wordmap);
        }
        for(int i = 0; i < hiddenConfig.size(); i++){
            gradientChecking_LSTMLayer(HiddenLayers, smr, sampleX, sampleY, i);   
        }
         gradientChecking_SoftmaxLayer(HiddenLayers, smr, sampleX, sampleY);
    }else{
        
        cout<<"****************************************************************************"<<endl
            <<"**                       TRAINING NETWORK......                             "<<endl
            <<"****************************************************************************"<<endl<<endl;
       
        softmaxUpdater smud(smr);
        LSTMLayerUpdater LSTMud(HiddenLayers);
        int k = 0;
        for(int epo = 1; epo <= training_epochs; epo++){
            for(; k <= iter_per_epo * epo; k++){
                cout<<"epoch: "<<epo<<", iter: "<<k;//<<endl;     
                std::vector<Mat> sampleX;
                Mat sampleY = Mat::zeros(nGram, batch_size, CV_64FC1);
                if(use_word2vec){
                    getSample(x, sampleX, y, sampleY, re_wordmap, wordvec);
                }else{
                    getSample(x, sampleX, y, sampleY, re_wordmap);
                }
                if(getNetworkCost(sampleX, sampleY, HiddenLayers, smr)){
                    // softmax update
                    smud.update(smr, k);
                    // hidden layer update
                    LSTMud.update(HiddenLayers, k);
                }
                sampleX.clear();
                std::vector<Mat>().swap(sampleX);
            }
            if(!is_gradient_checking){
                
                cout<<"Test training data: "<<endl;;
                testNetwork(x, y, HiddenLayers, smr, re_wordmap, wordvec);
                cout<<"Test testing data: "<<endl;;
                testNetwork(tx, ty, HiddenLayers, smr, re_wordmap, wordvec);   
                if(use_log){
                    save2XML("log", i2str(k), HiddenLayers, smr, re_wordmap);
                }
                //*/
            }
        }
    }
}

