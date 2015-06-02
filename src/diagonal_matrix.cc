#include "diagonal_matrix.h"

using namespace std;

diagonalMatrix::diagonalMatrix() {}
diagonalMatrix::diagonalMatrix(Mat& M){
    init(M);
}
diagonalMatrix::diagonalMatrix(int len){
    init(len);
}

diagonalMatrix diagonalMatrix::operator=(diagonalMatrix dm){
    return dm;
}

diagonalMatrix::~diagonalMatrix(){
    diagonal.release();
    full.release();
}

Mat diagonalMatrix::operator*(Mat &m){
    return full * m;
}

Mat diagonalMatrix::operator+(Mat &m){
    return full + m;
}

Mat diagonalMatrix::operator-(Mat &m){
    return full - m;
}

diagonalMatrix diagonalMatrix::operator+(diagonalMatrix dm){
    dm.diagonal += diagonal;
    dm.full = Mat::diag(dm.diagonal);
    return dm;
}

diagonalMatrix diagonalMatrix::operator-(diagonalMatrix dm){
    dm.diagonal -= diagonal;
    dm.full = Mat::diag(dm.diagonal);
    return dm;
}

void diagonalMatrix::mul(Mat &m){
    full = full.Mat::mul(m);
    diagonal = full.diag(0);
}

void diagonalMatrix::mul(diagonalMatrix &dm){
    mul(dm.full);
}

void diagonalMatrix::copyFrom(diagonalMatrix &dm){
    init(dm.full);
}

void diagonalMatrix::setSize(int length){
    size = length;
}

Mat diagonalMatrix::getDiagonal(){
    return diagonal;
}

Mat diagonalMatrix::getFull(){
    return full;
}

int diagonalMatrix::getSize(){
    return size;
}

void diagonalMatrix::update(int mode){
    if(mode == UPDATE_FROM_DIAG){
        full = Mat::diag(diagonal);
    }else{ // mode == UPDATE_FROM_FULL
        diagonal = full.diag(0);
    }
}

void diagonalMatrix::init(int length){
    setSize(length);
    diagonal = Mat::zeros(size, 1, CV_64FC1);
    full = Mat::zeros(size, size, CV_64FC1);
}

void diagonalMatrix::init(Mat& M){
    setSize(M.rows);
    if(M.cols == 1){
        M.copyTo(diagonal);
    }else{
        diagonal = M.diag(0);
    }
    full = Mat::diag(diagonal);
}

void diagonalMatrix::randomInit(int length, double epsilon){
    setSize(length);
    diagonal = Mat::ones(size, 1, CV_64FC1);
    randu(diagonal, Scalar(-1.0), Scalar(1.0));
    diagonal = diagonal * epsilon;
    full = Mat::diag(diagonal);
}
