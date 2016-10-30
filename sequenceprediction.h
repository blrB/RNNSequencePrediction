#ifndef SEQUENCEPREDICTION
#define SEQUENCEPREDICTION

#include <iostream>
#include <vector>
#include <armadillo>

#include "learningvector.h"

using namespace std;
using namespace arma;

class SequencePrediction
{
private:
    int n;
    int m;
    double e;
    double a;
    int maxIteration;
    mat W;
    mat V;
    vector<LearningVector> learningVectors;
    double contextNeuron;
    int L;
    int predict;
    double T;
    mat Ti;
    double getErrorDegree(double delta);
    void createWeightMatrix();
    void makeSequence();
    double getContextNeuron();
    void setContextNeuron(double neuron);
    void predictNo();
    void debug(mat X, mat P, mat Y, mat W, mat V, double answer);
public:
    SequencePrediction();
    void initANN();
    void run();
};

#endif // SEQUENCEPREDICTION

