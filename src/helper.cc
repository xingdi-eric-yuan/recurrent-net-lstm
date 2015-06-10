#include "helper.h"

using namespace std;

// int to string
string i2str(int num){
    stringstream ss;
    ss<<num;
    string s = ss.str();
    return s;
}

// string to int
int str2i(string str){
    return atoi(str.c_str());
}

Mat 
vec2Mat(const std::vector<int> &labelvec){
    Mat res = Mat::zeros(1, labelvec.size(), CV_64FC1);
    for(int i = 0; i < labelvec.size(); i++){
        res.ATD(0, i) = (double)(labelvec[i]);
    }
    return res;
}

bool 
isNumber(std::string &str){
    if(str.empty()) return false;
    for(int i = 0; i < str.size(); i++){
        if(str[i] < '0' || str[i] > '9') return false;
    }
    return true;
}

void 
removeNumber(std::vector<std::vector<singleWord> >& data){
    for(int i = 0; i < data.size(); i++){
        for(int j = 0; j < data[i].size(); j++){
            if(isNumber(data[i][j].word)) data[i][j].word = "___DIGIT___";
        }
    }
}

void 
getWordMap(const std::vector<std::vector<singleWord> >& data, std::unordered_map<string, int> &map, std::vector<string> &re_map){

    map.clear();
    re_map.clear();
    for(int i = 0; i < data.size(); i++){
        for(int j = 0; j < data[i].size(); j++){
            if(map.find(data[i][j].word) == map.end()){
                map[data[i][j].word] = re_map.size();
                re_map.push_back(data[i][j].word);
            }
        }
    }
    map["___UNDEFINED___"] = re_map.size();
    re_map.push_back("___UNDEFINED___");
    map["___PADDING___"] = re_map.size();
    re_map.push_back("___PADDING___");
}

int 
label2num(std::string label){
    int res = 0;
    if(label.compare("O") == 0){
        res = 0;
    }elif(label.compare("B-NEWSTYPE") == 0){
        res = 1;
    }elif(label.compare("B-PROVIDER") == 0){
        res = 2;
    }elif(label.compare("B-KEYWORDS") == 0){
        res = 3;
    }elif(label.compare("B-SECTION") == 0){
        res = 4;
    }elif(label.compare("I-NEWSTYPE") == 0){
        res = 5;
    }elif(label.compare("I-PROVIDER") == 0){
        res = 6;
    }elif(label.compare("I-KEYWORDS") == 0){
        res = 7;
    }elif(label.compare("I-SECTION") == 0){
        res = 8;
    }else{
        res = 9;
    }
    return res;
}

string 
num2label(int num){
    string res = "";
    if(num == 0){
        res = "O";
    }elif(num == 1){
        res = "B-NEWSTYPE";
    }elif(num == 2){
        res = "B-PROVIDER";
    }elif(num == 3){
        res = "B-KEYWORDS";
    }elif(num == 4){
        res = "B-SECTION";
    }elif(num == 5){
        res = "I-NEWSTYPE";
    }elif(num == 6){
        res = "I-PROVIDER";
    }elif(num == 7){
        res = "I-KEYWORDS";
    }elif(num == 8){
        res = "I-SECTION";
    }elif(num == 9){
        res = "ERROR";
    }
    return res;
}

void 
breakString(string str, std::vector<string> &vec){
    vec.clear();
    int head = 0;
    int tail = 0;
    while(true){
        if(head >= str.length()) break;
        if(str[tail] == ','){
            vec.push_back(str.substr(head, tail - head));
            head = tail + 1;
            tail = head;
        }else ++ tail;
    }
}

Mat 
oneOfN(int one, int n){
    Mat res = Mat::zeros(n, 1, CV_64FC1);
    res.ATD(one, 0) = 1.0;
    return res;
}

/*
void 
getSample(const std::vector<std::vector<int> >& src1, std::vector<Mat>& dst1, const std::vector<std::vector<int> >& src2, Mat& dst2, std::vector<string> &re_wordmap){
    dst1.clear();
    int _size = dst2.cols;
    int T = src1[0].size();
    for(int i = 0; i < T; i++){
        Mat tmp = Mat::zeros(re_wordmap.size(), _size, CV_64FC1);
        dst1.push_back(tmp);
    }

    random_shuffle(sample_vec.begin(), sample_vec.end());
    for(int i = 0; i < _size; i++){
        int randomNum = sample_vec[i];
        for(int j = 0; j < T; j++){
            Mat tmp1 = oneOfN(src1[randomNum][j], re_wordmap.size());
            Rect roi = Rect(i, 0, 1, re_wordmap.size());
            Mat tmp2 = dst1[j](roi);
            tmp1.copyTo(tmp2);
        }
        dst2.ATD(0, i) = src2[randomNum][T - 1];
    }
}
*/

void 
getSample(const std::vector<std::vector<int> >& src1, std::vector<Mat>& dst1, const std::vector<std::vector<int> >& src2, Mat& dst2, std::vector<string> &re_wordmap){
    dst1.clear();
    int _size = dst2.cols;
    int T = src1[0].size();
    for(int i = 0; i < T; i++){
        Mat tmp = Mat::zeros(re_wordmap.size(), _size, CV_64FC1);
        dst1.push_back(tmp);
    }
    random_shuffle(sample_vec.begin(), sample_vec.end());
    for(int i = 0; i < _size; i++){
        int randomNum = sample_vec[i];
        for(int j = 0; j < T; j++){
            Mat tmp1 = oneOfN(src1[randomNum][j], re_wordmap.size());
            Rect roi = Rect(i, 0, 1, re_wordmap.size());
            Mat tmp2 = dst1[j](roi);
            tmp1.copyTo(tmp2);
            dst2.ATD(j, i) = src2[randomNum][j];
        }
    }
}

void 
getSample(const std::vector<std::vector<int> >& src1, std::vector<Mat>& dst1, const std::vector<std::vector<int> >& src2, Mat& dst2, std::vector<string> &re_wordmap, std::unordered_map<std::string, Mat> &wordvec){
    dst1.clear();
    int _size = dst2.cols;
    int T = src1[0].size();
    for(int i = 0; i < T; i++){
        Mat tmp = Mat::zeros(word_vec_len, _size, CV_64FC1);
        dst1.push_back(tmp);
    }
    random_shuffle(sample_vec.begin(), sample_vec.end());
    int fail = 0;
    for(int i = 0; i - fail < _size; i++){
        int randomNum = sample_vec[i];
        for(int j = 0; j < T; j++){
            Mat tmp1;
            if(wordvec.find(re_wordmap[src1[randomNum][j]]) == wordvec.end()) {++ fail; break;}
            wordvec[re_wordmap[src1[randomNum][j]]].copyTo(tmp1);
            Rect roi = Rect(i - fail, 0, 1, word_vec_len);
            Mat tmp2 = dst1[j](roi);
            tmp1.copyTo(tmp2);
            dst2.ATD(j, i - fail) = src2[randomNum][j];
        }
    }
}

void 
getDataMat(const std::vector<std::vector<int> >& src, std::vector<Mat>& dst, std::vector<string> &re_wordmap){
    dst.clear();
    int _size = src.size();
    int T = src[0].size();
    for(int i = 0; i < T; i++){
        Mat tmp = Mat::zeros(re_wordmap.size(), _size, CV_64FC1);
        dst.push_back(tmp);
    }
    for(int i = 0; i < _size; i++){
        for(int j = 0; j < T; j++){
            Mat tmp1 = oneOfN(src[i][j], re_wordmap.size());
            Rect roi = Rect(i, 0, 1, re_wordmap.size());
            Mat tmp2 = dst[j](roi);
            tmp1.copyTo(tmp2);
        }
    }
}

void 
getDataMat(const std::vector<std::vector<int> >& src, std::vector<Mat>& dst, std::vector<string> &re_wordmap, std::unordered_map<string, Mat> &wordvec){
    dst.clear();
    int _size = src.size();
    int T = src[0].size();
    for(int i = 0; i < T; i++){
        Mat tmp = Mat::zeros(word_vec_len, _size, CV_64FC1);
        dst.push_back(tmp);
    }
    for(int i = 0; i < _size; i++){
        for(int j = 0; j < T; j++){
            Mat tmp1;
            wordvec[re_wordmap[src[i][j]]].copyTo(tmp1);
            Rect roi = Rect(i, 0, 1, word_vec_len);
            Mat tmp2 = dst[j](roi);
            tmp1.copyTo(tmp2);
        }
    }
}

void 
getLabelMat(const std::vector<std::vector<int> >& src, Mat& dst){
    int _size = dst.cols;
    int T = src[0].size();
    int mid = (int)(T /2.0);
    for(int i = 0; i < _size; i++){
        dst.ATD(0, i) = src[i][mid];
    }
}
