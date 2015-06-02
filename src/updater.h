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
    Mat velocity_W;
    Mat second_derivative_W;
    Mat learning_rate;
};

class recurrentLayerUpdater {//: public updater{
public:
	recurrentLayerUpdater(std::vector<Rl>&);
	~recurrentLayerUpdater();
	void init(std::vector<Rl>&);
	void setMomentum();
	void update(std::vector<Rl>&, int);

	double momentum_W;
	double momentum_second_derivative;
	int iter;
	double mu;
    std::vector<Mat> velocity_W;
    std::vector<Mat>  velocity_U;
    std::vector<Mat>  second_derivative_W;
    std::vector<Mat>  second_derivative_U;
    Mat learning_rate_W;
    Mat learning_rate_U;
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

	std::vector<Mat> velocity_W_input;
	std::vector<Mat> velocity_W_forget;
	std::vector<Mat> velocity_W_cell;
	std::vector<Mat> velocity_W_output;
    
    std::vector<Mat>  velocity_U_input;
    std::vector<Mat>  velocity_U_forget;
    std::vector<Mat>  velocity_U_cell;
    std::vector<Mat>  velocity_U_output;

    std::vector<diagonalMatrix*> velocity_V_input;
    std::vector<diagonalMatrix*> velocity_V_forget;
    std::vector<diagonalMatrix*> velocity_V_output;

	std::vector<Mat> second_derivative_W_input;
	std::vector<Mat> second_derivative_W_forget;
	std::vector<Mat> second_derivative_W_cell;
	std::vector<Mat> second_derivative_W_output;
    
    std::vector<Mat>  second_derivative_U_input;
    std::vector<Mat>  second_derivative_U_forget;
    std::vector<Mat>  second_derivative_U_cell;
    std::vector<Mat>  second_derivative_U_output;

    std::vector<diagonalMatrix*> second_derivative_V_input;
    std::vector<diagonalMatrix*> second_derivative_V_forget;
    std::vector<diagonalMatrix*> second_derivative_V_output;

    Mat learning_rate_W;
    Mat learning_rate_U;
    Mat learning_rate_V;
};






