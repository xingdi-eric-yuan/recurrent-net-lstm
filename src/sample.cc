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
bool use_word2vec = false;
int batch_size = 1;
int log_iter = 0;
int io_non_linearity = 2;
int gate_non_linearity = 2;
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
    //readDataset("dataset/CoNLL04/trainData.txt", "dataset/CoNLL04/ne.train.pred",
    //            "dataset/CoNLL04/testData.txt", "dataset/CoNLL04/ne.test.pred", 
    //            trainData, testData, labelmap, re_labelmap);

    cout<<"Successfully read dataset, there're "<<trainData.size()<<" sentences in training set, and "<<testData.size()<<" sentences in test set."<<endl;
/*
// for word2vec
    ofstream fout;
    fout.open("trainData.txt");
    for(int i = 0; i < trainData.size(); i++){
        for(int j = 0; j < trainData[i].size(); j++){
            fout<<trainData[i][j].word<<endl;
            if(j == trainData[i].size() - 1) fout<<endl;
        }
    }
    fout.close();

//*/

    softmaxConfig.NumClasses = labelmap.size();

    // change all number-word into "__DIGIT__"
    removeNumber(trainData);
    // get word map:
    // for each unique word in dataset, give it a unique id
    std::unordered_map<string, int> wordmap;
    std::vector<string> re_wordmap;
    getWordMap(trainData, wordmap, re_wordmap);
    std::vector<std::vector<int> > trainX;
    std::vector<std::vector<int> > trainY;
    std::vector<std::vector<int> > testX;
    std::vector<std::vector<int> > testY;
    unordered_map<string, Mat> wordvec;
    if(use_word2vec){
        word_vec_len = 300;
        readWordvec("dataset/wordvecs.txt", wordvec);
        //readWordvec("dataset/CoNLL04/CoNLL04wordvecs.txt", wordvec);
        cout<<"Successfully read wordvecs, map size is "<<wordvec.size()<<endl;
        cout<<"The dimension of network input is "<<word_vec_len<<endl;
    }else{
        // For 1 of n encoding method, the input size of rnn is the size
        // of wordmap (one "1" and all others are "0")
        word_vec_len = re_wordmap.size();
        cout<<"The dimension of network input is "<<word_vec_len<<endl;
    }

    // Break sentences into sub-sentences have length of nGram,
    // padding method is used
    resolutioner(trainData, trainX, trainY, wordmap);
    cout<<"there are "<<labelmap.size()<<" classes of labels..."<<endl;
    cout<<"there are "<<trainX.size()<<" sub-sentences in training data..."<<endl;
    resolutioner(testData, testX, testY, wordmap);
    cout<<"there are "<<testX.size()<<" sub-sentences in test data..."<<endl;

    int nsamples = trainX.size();
    for(int i = 0; i < nsamples; i++){
        sample_vec.push_back(i);
    }
    std::vector<LSTMl> HiddenLayers;
    Smr smr;
    rnnInitPrarms(HiddenLayers, smr);
    // Train network using Back Propogation
    trainNetwork(trainX, trainY, HiddenLayers, smr, testX, testY, re_wordmap, wordvec);
    
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







