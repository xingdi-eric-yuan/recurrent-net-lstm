#include "matrix_maths.h"

using namespace cv;
using namespace std;

double 
Reciprocal(const double &s){
    double res = 1.0;
    res /= s;
    return res;
}

Mat 
Reciprocal(const Mat &M){
    return div(1.0, M);
}

Mat 
sigmoid(const Mat &M){
    return div(1.0, (exp(-M) + 1.0));
//    return 1.0 / (exp(-M) + 1.0);
}

Mat 
dsigmoid_a(const Mat &a){
    Mat res = 1.0 - a;
    res = res.mul(a);
    return res;
}

Mat 
dsigmoid(const Mat &M){
    return divide(exp(M), pow((1.0 + exp(M)), 2.0));
}

Mat
ReLU(const Mat& M){
    Mat res = M > 0.0;
    res.convertTo(res, CV_64FC1, 1.0 / 255, 0);
    res = res.mul(M);
    return res;
}

Mat
dReLU(const Mat& M){
    Mat res = M > 0.0;
    res.convertTo(res, CV_64FC1, 1.0 / 255.0);
    return res;
}

Mat 
Tanh(const Mat &M){
    Mat res;
    M.copyTo(res);
    for(int i=0; i<res.rows; i++){
        for(int j=0; j<res.cols; j++){
            res.ATD(i, j) = tanh(M.ATD(i, j));
        }
    }
    return res;
}

Mat
dTanh(const Mat &M){
    Mat res = Mat::ones(M.rows, M.cols, CV_64FC1);
    return res - M.mul(M);
}

Mat 
nonLinearity(const Mat &M, int type){
    int tmpnl = 0;
    if(type == IO_NL) tmpnl = io_non_linearity;
    else tmpnl = gate_non_linearity; // type == GATE_NL
    if(tmpnl == NL_RELU){
        return ReLU(M);
    }elif(tmpnl == NL_TANH){
        return Tanh(M);
    }else{
        return sigmoid(M);
    }
}

Mat 
dnonLinearity(const Mat &M, int type){
    int tmpnl = 0;
    if(type == IO_NL) tmpnl = io_non_linearity;
    else tmpnl = gate_non_linearity; // type == GATE_NL
    if(tmpnl == NL_RELU){
        return dReLU(M);
    }elif(tmpnl == NL_TANH){
        return dTanh(M);
    }else{
        return dsigmoid(M);
    }
}

// Mimic rot90() in Matlab/GNU Octave.
Mat 
rot90(const Mat &M, int k){
    Mat res;
    if(k == 0) return M;
    elif(k == 1){
        flip(M.t(), res, 0);
    }else{
        flip(rot90(M, k - 1).t(), res, 0);
    }
    return res;
}

// get KroneckerProduct 
// for upsample
// see function kron() in Matlab/Octave
Mat
kron(const Mat &a, const Mat &b){
    Mat res = Mat::zeros(a.rows * b.rows, a.cols * b.cols, CV_64FC1);
    for(int i=0; i<a.rows; i++){
        for(int j=0; j<a.cols; j++){
            Rect roi = Rect(j * b.cols, i * b.rows, b.cols, b.rows);
            Mat temp = res(roi);
            Mat c = b.mul(a.ATD(i, j));
            c.copyTo(temp);
        }
    }
    return res;
}

Mat 
getBernoulliMatrix(int height, int width, double prob){
    // randu builds a Uniformly distributed matrix
    Mat ran = Mat::zeros(height, width, CV_64FC1);
    randu(ran, Scalar(0), Scalar(1.0));
    Mat res = ran <= prob;
    res.convertTo(res, CV_64FC1, 1.0 / 255, 0);
    ran.release();
    return res;
}

// Follows are OpenCV maths
Mat 
exp(const Mat &src){
    Mat dst;
    exp(src, dst);
    return dst;
}

Mat 
div(double x, const Mat &src){
    Mat dst;
    src.copyTo(dst);
    for(int i = 0; i < dst.rows; i++){
        for(int j = 0; j < dst.cols; j++){
            if(dst.ATD(i, j) != 0.0) dst.ATD(i, j) = x / dst.ATD(i, j);
        }
    }
    return dst;
}

Mat 
log(const Mat &src){
    Mat dst;
    log(src, dst);
    return dst;
}

Mat 
reduce(const Mat &src, int direc, int conf){
    Mat dst;
    reduce(src, dst, direc, conf);
    return dst;
}

Mat 
divide(const Mat &m1, const Mat &m2){
    Mat dst;
    divide(m1, m2, dst);
    return dst;
}

Mat 
pow(const Mat &m1, double val){
    Mat dst;
    pow(m1, val, dst);
    return dst;
}

double 
sum1(const Mat &m){
    Scalar tmp = sum(m);
    return tmp[0];
}

double
max(const Mat &m){
    Point min;
    Point max;
    double minval;
    double maxval;
    minMaxLoc(m, &minval, &maxval, &min, &max);
    return maxval;
}

double
min(const Mat &m){
    Point min;
    Point max;
    double minval;
    double maxval;
    minMaxLoc(m, &minval, &maxval, &min, &max);
    return minval;
}








