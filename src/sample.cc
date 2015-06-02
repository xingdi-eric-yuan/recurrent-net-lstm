#include "general_settings.h"
using namespace std;
using namespace cv;

std::vector<HiddenLayerConfig> hiddenConfig;
SoftmaxLayerConfig softmaxConfig;
std::vector<int> sample_vec;
///////////////////////////////////
// General parameters init before 
// reading config file
///////////////////////////////////
bool is_gradient_checking = false;
bool use_log = false;
int batch_size = 1;
int log_iter = 0;
int non_linearity = 2;
int training_epochs = 0;
double lrate_w = 0.0;
double momentum_w_init = 0.5;
double momentum_d2_init = 0.5;
double momentum_w_adjust = 0.95;
double momentum_d2_adjust = 0.90;

int iter_per_epo = 0;
int word_vec_len = 0;
int nGram = 3;
float training_percent = 0.8;
double prev_cost = -1.0;

void 
run(){
    long start, end;
    start = clock();

    readConfigFile("config.txt", true);
    std::vector<std::vector<singleWord> > trainData;
    std::vector<std::vector<singleWord> > testData;
    std::unordered_map<string, int> labelmap;
    std::vector<string> re_labelmap;
    readDataset("dataset/news_tagged_data.txt", trainData, testData, labelmap, re_labelmap);
    cout<<"Successfully read dataset, training data size is "<<trainData.size()<<", test data size is "<<testData.size()<<endl;
    softmaxConfig.NumClasses = labelmap.size();

    // change all number-word into "__DIGIT__"
    removeNumber(trainData);
    // get word map:
    // for each unique word in dataset, give it a unique id
    std::unordered_map<string, int> wordmap;
    std::vector<string> re_wordmap;
    getWordMap(trainData, wordmap, re_wordmap);
    // For 1 of n encoding method, the input size of rnn is the size
    // of wordmap (one "1" and all others are "0")
    word_vec_len = re_wordmap.size();

    std::vector<std::vector<int> > trainX;
    std::vector<std::vector<int> > trainY;
    // Break sentences into sub-sentences have length of nGram,
    // padding method is used
    resolutioner(trainData, trainX, trainY, wordmap);
    cout<<"there are "<<trainX.size()<<" training data..."<<endl;
    cout<<"there are "<<labelmap.size()<<" kind of labels..."<<endl;

    std::vector<std::vector<int> > testX;
    std::vector<std::vector<int> > testY;
    resolutioner(testData, testX, testY, wordmap);
    cout<<"there are "<<testX.size()<<" test data..."<<endl;

    int nsamples = trainX.size();
    for(int i = 0; i < nsamples; i++){
        sample_vec.push_back(i);
    }
    std::vector<LSTMl> HiddenLayers;
    Smr smr;
    rnnInitPrarms(HiddenLayers, smr);

    // Train network using Back Propogation
    trainNetwork(trainX, trainY, HiddenLayers, smr, testX, testY, re_wordmap);
    
    HiddenLayers.clear();
    trainData.clear();
    std::vector<std::vector<singleWord> >().swap(trainData);
    testData.clear();
    std::vector<std::vector<singleWord> >().swap(trainData);
    end = clock();
    cout<<"Totally used time: "<<((double)(end - start)) / CLOCKS_PER_SEC<<" second"<<endl;
}

int 
main(int argc, char** argv){

    run();

    return 0;
}







