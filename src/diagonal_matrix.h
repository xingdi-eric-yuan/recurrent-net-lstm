#pragma once
#include "general_settings.h"

using namespace std;
using namespace cv;


    class diagonalMatrix{
    public:

        diagonalMatrix();
        diagonalMatrix(Mat&);
        diagonalMatrix(int);
        diagonalMatrix operator=(diagonalMatrix);
        ~diagonalMatrix();
        Mat operator*(Mat&);
        Mat operator+(Mat&);
        Mat operator-(Mat&);
        diagonalMatrix operator+(diagonalMatrix);
        diagonalMatrix operator-(diagonalMatrix);
        void mul(Mat&);
        void mul(diagonalMatrix&);
        void copyFrom(diagonalMatrix&);
        void setSize(int);
        void update(int);
        Mat getDiagonal();
        Mat getFull();
        void init(int);
        void init(Mat&);
        void randomInit(int, double);
        int getSize();

    //private:
        int size;
        Mat diagonal;
        Mat full;
    };

