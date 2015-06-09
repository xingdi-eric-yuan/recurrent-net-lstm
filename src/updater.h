#pragma once
#include "general_settings.h"
using namespace std;

class updater{
public:
	updater();
	virtual ~updater();
};

class softmaxUpdater {//: public updater{
public:
	softmaxUpdater(Smr&);
	~softmaxUpdater();
	void init(Smr&);
	void setMomentum();
	void update(Smr&, int);

	double momentum_W;
	double momentum_second_derivative;
	int iter;
	double mu;
    Mat velocity_W_left;
    Mat velocity_W_right;
    Mat second_derivative_W_left;
    Mat second_derivative_W_right;
    Mat learning_rate;
};

class LSTMLayerUpdater {//: public updater{
public:
	LSTMLayerUpdater(std::vector<LSTMl>&);
	~LSTMLayerUpdater();
	void init(std::vector<LSTMl>&);
	void setMomentum();
	void update(std::vector<LSTMl>&, int);

	double momentum_W;
	double momentum_second_derivative;
	int iter;
	double mu;

    std::vector<Mat> velocity_W_input_left;
    std::vector<Mat> velocity_W_forget_left;
    std::vector<Mat> velocity_W_cell_left;
    std::vector<Mat> velocity_W_output_left;
    
    std::vector<Mat>  velocity_U_input_left;
    std::vector<Mat>  velocity_U_forget_left;
    std::vector<Mat>  velocity_U_cell_left;
    std::vector<Mat>  velocity_U_output_left;

    std::vector<diagonalMatrix*> velocity_V_input_left;
    std::vector<diagonalMatrix*> velocity_V_forget_left;
    std::vector<diagonalMatrix*> velocity_V_output_left;

    std::vector<Mat> second_derivative_W_input_left;
    std::vector<Mat> second_derivative_W_forget_left;
    std::vector<Mat> second_derivative_W_cell_left;
    std::vector<Mat> second_derivative_W_output_left;
    
    std::vector<Mat>  second_derivative_U_input_left;
    std::vector<Mat>  second_derivative_U_forget_left;
    std::vector<Mat>  second_derivative_U_cell_left;
    std::vector<Mat>  second_derivative_U_output_left;

    std::vector<diagonalMatrix*> second_derivative_V_input_left;
    std::vector<diagonalMatrix*> second_derivative_V_forget_left;
    std::vector<diagonalMatrix*> second_derivative_V_output_left;

    std::vector<Mat> velocity_W_input_right;
    std::vector<Mat> velocity_W_forget_right;
    std::vector<Mat> velocity_W_cell_right;
    std::vector<Mat> velocity_W_output_right;
    
    std::vector<Mat>  velocity_U_input_right;
    std::vector<Mat>  velocity_U_forget_right;
    std::vector<Mat>  velocity_U_cell_right;
    std::vector<Mat>  velocity_U_output_right;

    std::vector<diagonalMatrix*> velocity_V_input_right;
    std::vector<diagonalMatrix*> velocity_V_forget_right;
    std::vector<diagonalMatrix*> velocity_V_output_right;

    std::vector<Mat> second_derivative_W_input_right;
    std::vector<Mat> second_derivative_W_forget_right;
    std::vector<Mat> second_derivative_W_cell_right;
    std::vector<Mat> second_derivative_W_output_right;
    
    std::vector<Mat>  second_derivative_U_input_right;
    std::vector<Mat>  second_derivative_U_forget_right;
    std::vector<Mat>  second_derivative_U_cell_right;
    std::vector<Mat>  second_derivative_U_output_right;

    std::vector<diagonalMatrix*> second_derivative_V_input_right;
    std::vector<diagonalMatrix*> second_derivative_V_forget_right;
    std::vector<diagonalMatrix*> second_derivative_V_output_right;

    Mat learning_rate_W;
    Mat learning_rate_U;
    Mat learning_rate_V;
};

//*/
