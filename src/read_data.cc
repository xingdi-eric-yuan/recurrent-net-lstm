#include "read_data.h"

using namespace std;
using namespace cv;


/////////////label map!!!!
void
readDataset(std::string path, 
    std::vector<std::vector<singleWord> >& trainData, 
    std::vector<std::vector<singleWord> >& testData, 
    std::unordered_map<string, int> &labelmap, 
    std::vector<string> &re_labelmap){
    std::vector<std::vector<singleWord> > data;
    ifstream infile(path);
    string line;
    std::vector<singleWord> sentence;
    labelmap["___PADDING___"] = 0;
    re_labelmap.push_back("___PADDING___");
    int counter = 1;
    while (getline(infile, line)){
        if(line.empty() || line[0] == ' '){
            if(!sentence.empty()){
                data.push_back(sentence);
                sentence.clear();
            }
        }else{
            istringstream iss(line);
            string tmpword;
            string tmplabel;
            iss >> tmpword >> tmplabel;
            if(labelmap.find(tmplabel) == labelmap.end()){
                labelmap[tmplabel] = counter++;
                re_labelmap.push_back(tmplabel);
            }
            singleWord tmpsw(tmpword, labelmap[tmplabel]);
            sentence.push_back(tmpsw);
        }
    }
    if(!sentence.empty()){
        data.push_back(sentence);
        sentence.clear();
    }
    // random shuffle
    random_shuffle(data.begin(), data.end());
    // cross validation
    int trainSize = (int)((float)data.size() * training_percent);
    for(int i = 0; i < data.size(); ++i){
        if(i < trainSize) trainData.push_back(data[i]);
        else testData.push_back(data[i]);
    }
    data.clear();
    std::vector<std::vector<singleWord> >().swap(data);
}

// read data from stdin
void 
readLine(std::vector<string> &str){
    string line;
    getline(cin, line);

    istringstream stm(line);
    string word;
    while(stm >> word) {
        str.push_back(word);
    }
}

void 
resolutioner(const std::vector<std::vector<singleWord> > &data, std::vector<std::vector<int> > &resol, std::vector<std::vector<int> > &labels, std::unordered_map<string, int> &wordmap){

    std::vector<singleWord> tmpvec;
    std::vector<int> tmpresol;
    std::vector<int> tmplabel;
    for(int i = 0; i < data.size(); i++){
        tmpvec.clear();
        tmpvec = data[i];
        singleWord tmpsw("___PADDING___", 0);
        for(int j = 0; j < nGram - 1; j++){
            tmpvec.insert(tmpvec.begin(), tmpsw);
        }
        for(int j = 0; j < tmpvec.size() - nGram + 1; j++){
            tmpresol.clear();
            tmplabel.clear();
            for(int k = 0; k < nGram; k++){
                if(wordmap.find(tmpvec[j + k].word) == wordmap.end()){
                    tmpresol.push_back(wordmap["___UNDEFINED___"]);
                }else{
                    tmpresol.push_back(wordmap[tmpvec[j + k].word]);
                }
                tmplabel.push_back(tmpvec[j + k].label);
            }
            resol.push_back(tmpresol);
            labels.push_back(tmplabel);
        }
    }
}
