#include "weights_IO.h"
using namespace cv;
using namespace std;
/*
void 
save2txt(const Mat &data, string path, string str){
    string tmp = path + str;
    FILE *pOut = fopen(tmp.c_str(), "w");
    for(int i = 0; i < data.rows; i++){
        for(int j = 0; j < data.cols; j++){
            fprintf(pOut, "%lf", data.ATD(i, j));
            if(j == data.cols - 1){
                fprintf(pOut, "\n");
            } 
            else{
                fprintf(pOut, " ");
            } 
        }
    }
    fclose(pOut);
}

void 
save2XML(string path, string name, const std::vector<Hl> &Hiddenlayers, const Smr &smr, const std::vector<string> &re_resolmap){

    mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    string tmp = path + "/" + name + ".xml";
    FileStorage fs(tmp, FileStorage::WRITE);
    
    fs << "smr_W" << smr.W;

    for(int i = 0; i < Hiddenlayers.size(); i++){
        tmp = "hlayer" + std::to_string(i);
        fs << (tmp + "_W") << Hiddenlayers[i].W;
        fs << (tmp + "_U") << Hiddenlayers[i].U;
    }
    fs << "re_resolmap" << re_resolmap;
    fs.release();
    cout<<"Successfully saved network information..."<<endl;
}

void 
readFromXML(string path, std::vector<Hl> &Hiddenlayers, Smr &smr, std::vector<string> &re_resolmap){

    string tmp = "";
    FileStorage fs(path, FileStorage::READ);
    fs["smr_W"] >> smr.W;
    for(int i = 0; i < Hiddenlayers.size(); i++){
        tmp = "hlayer" + std::to_string(i);
        fs[tmp + "_W"] >> Hiddenlayers[i].W;
        fs[tmp + "_U"] >> Hiddenlayers[i].U;
    }
    fs["re_resolmap"] >> re_resolmap;
    fs.release();
    cout<<"Successfully read network information from "<<path<<"..."<<endl;
}
*/
