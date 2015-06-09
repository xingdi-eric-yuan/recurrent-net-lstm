#include "weights_IO.h"
using namespace cv;
using namespace std;

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
save2XML(string path, string name, const std::vector<LSTMl> &Hiddenlayers, const Smr &smr, const std::vector<string> &re_resolmap){

    mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    string tmp = path + "/" + name + ".xml";
    FileStorage fs(tmp, FileStorage::WRITE);
    
    fs << "smr_W_left" << smr.W_left;
    fs << "smr_W_right" << smr.W_right;

    for(int i = 0; i < Hiddenlayers.size(); i++){
        tmp = "hlayer" + std::to_string(i);
        fs << (tmp + "_W_input_left") << Hiddenlayers[i].W_input_left;
        fs << (tmp + "_W_forget_left") << Hiddenlayers[i].W_forget_left;
        fs << (tmp + "_W_cell_left") << Hiddenlayers[i].W_cell_left;
        fs << (tmp + "_W_output_left") << Hiddenlayers[i].W_output_left;
        fs << (tmp + "_U_input_left") << Hiddenlayers[i].U_input_left;
        fs << (tmp + "_U_forget_left") << Hiddenlayers[i].U_forget_left;
        fs << (tmp + "_U_cell_left") << Hiddenlayers[i].U_cell_left;
        fs << (tmp + "_U_output_left") << Hiddenlayers[i].U_output_left;
        fs << (tmp + "_V_input_left") << Hiddenlayers[i].V_input_left -> diagonal;
        fs << (tmp + "_V_forget_left") << Hiddenlayers[i].V_forget_left -> diagonal;
        fs << (tmp + "_V_output_left") << Hiddenlayers[i].V_output_left -> diagonal;

        fs << (tmp + "_W_input_right") << Hiddenlayers[i].W_input_right;
        fs << (tmp + "_W_forget_right") << Hiddenlayers[i].W_forget_right;
        fs << (tmp + "_W_cell_right") << Hiddenlayers[i].W_cell_right;
        fs << (tmp + "_W_output_right") << Hiddenlayers[i].W_output_right;
        fs << (tmp + "_U_input_right") << Hiddenlayers[i].U_input_right;
        fs << (tmp + "_U_forget_right") << Hiddenlayers[i].U_forget_right;
        fs << (tmp + "_U_cell_right") << Hiddenlayers[i].U_cell_right;
        fs << (tmp + "_U_output_right") << Hiddenlayers[i].U_output_right;
        fs << (tmp + "_V_input_right") << Hiddenlayers[i].V_input_right -> diagonal;
        fs << (tmp + "_V_forget_right") << Hiddenlayers[i].V_forget_right -> diagonal;
        fs << (tmp + "_V_output_right") << Hiddenlayers[i].V_output_right -> diagonal;
    }
    fs << "re_resolmap" << re_resolmap;
    fs.release();
    cout<<"Successfully saved network information..."<<endl;
}

void 
readFromXML(string path, std::vector<LSTMl> &Hiddenlayers, Smr &smr, std::vector<string> &re_resolmap){

    string tmp = "";
    FileStorage fs(path, FileStorage::READ);
    fs["smr_W_left"] >> smr.W_left;
    fs["smr_W_right"] >> smr.W_right;
    for(int i = 0; i < Hiddenlayers.size(); i++){
        tmp = "hlayer" + std::to_string(i);
        fs[tmp + "_W_input_left"] >> Hiddenlayers[i].W_input_left;
        fs[tmp + "_W_forget_left"] >> Hiddenlayers[i].W_forget_left;
        fs[tmp + "_W_cell_left"] >> Hiddenlayers[i].W_cell_left;
        fs[tmp + "_W_output_left"] >> Hiddenlayers[i].W_output_left;
        fs[tmp + "_U_input_left"] >> Hiddenlayers[i].U_input_left;
        fs[tmp + "_U_forget_left"] >> Hiddenlayers[i].U_forget_left;
        fs[tmp + "_U_cell_left"] >> Hiddenlayers[i].U_cell_left;
        fs[tmp + "_U_output_left"] >> Hiddenlayers[i].U_output_left;
        fs[tmp + "_V_input_left"] >> Hiddenlayers[i].V_input_left -> diagonal;
        fs[tmp + "_V_forget_left"] >> Hiddenlayers[i].V_forget_left -> diagonal;
        fs[tmp + "_V_output_left"] >> Hiddenlayers[i].V_output_left -> diagonal;
        Hiddenlayers[i].V_input_left -> update(UPDATE_FROM_DIAG);
        Hiddenlayers[i].V_forget_left -> update(UPDATE_FROM_DIAG);
        Hiddenlayers[i].V_output_left -> update(UPDATE_FROM_DIAG);

        fs[tmp + "_W_input_right"] >> Hiddenlayers[i].W_input_right;
        fs[tmp + "_W_forget_right"] >> Hiddenlayers[i].W_forget_right;
        fs[tmp + "_W_cell_right"] >> Hiddenlayers[i].W_cell_right;
        fs[tmp + "_W_output_right"] >> Hiddenlayers[i].W_output_right;
        fs[tmp + "_U_input_right"] >> Hiddenlayers[i].U_input_right;
        fs[tmp + "_U_forget_right"] >> Hiddenlayers[i].U_forget_right;
        fs[tmp + "_U_cell_right"] >> Hiddenlayers[i].U_cell_right;
        fs[tmp + "_U_output_right"] >> Hiddenlayers[i].U_output_right;
        fs[tmp + "_V_input_right"] >> Hiddenlayers[i].V_input_right -> diagonal;
        fs[tmp + "_V_forget_right"] >> Hiddenlayers[i].V_forget_right -> diagonal;
        fs[tmp + "_V_output_right"] >> Hiddenlayers[i].V_output_right -> diagonal;
        Hiddenlayers[i].V_input_right -> update(UPDATE_FROM_DIAG);
        Hiddenlayers[i].V_forget_right -> update(UPDATE_FROM_DIAG);
        Hiddenlayers[i].V_output_right -> update(UPDATE_FROM_DIAG);
    }
    fs["re_resolmap"] >> re_resolmap;
    fs.release();
    cout<<"Successfully read network information from "<<path<<"..."<<endl;
}

